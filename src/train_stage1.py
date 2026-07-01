"""
Stage 1 training: multilingual message-level AGGRESSION classifier.

Fine-tunes m-BERT / MuRIL on data/processed/messages.csv (English + Roman Urdu)
to predict whether a single message is aggressive. Uses class weighting to
counter the aggressive/non-aggressive imbalance.

Uses a 3-way split: TRAIN (fit) / VAL (model selection) / TEST (final, untouched
report). The reported test metrics are written to models/eval_stage1_<tag>.txt
so they can be cited directly in the thesis.

Run locally:   python -m src.train_stage1
On Kaggle:     from src.train_stage1 import train_stage1; train_stage1(text_model='google/muril-base-cased')
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)
from transformers import AutoTokenizer

from src.config import (
    DEVICE, TEXT_MODEL, MESSAGES_CSV, MAX_LEN, RANDOM_SEED, MODEL_DIR
)
from src.data.message_dataset import MessageDataset
from src.models.aggression_model import AggressionClassifier


def _predict(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            tti = batch['token_type_ids'].to(device)
            y = batch['target'].to(device)
            logits = model(ids, mask, tti)
            total_loss += criterion(logits, y).item()
            preds.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())
    return total_loss / len(loader), np.concatenate(preds), np.concatenate(labels)


def _metrics(labels, probs, threshold=0.5):
    yhat = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, yhat)
    p, r, f1, _ = precision_recall_fscore_support(labels, yhat, average='binary', zero_division=0)
    return acc, p, r, f1, yhat


def train_stage1(csv_path=None, text_model=None, epochs=3, batch_size=16,
                 lr=2e-5, max_len=None, val_split=0.15, test_split=0.15):
    csv_path = csv_path or MESSAGES_CSV
    text_model = text_model or TEXT_MODEL
    max_len = max_len or MAX_LEN

    print(f"Device: {DEVICE}")
    print(f"Text model: {text_model}")
    print(f"Messages CSV: {csv_path}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Split: train/val/test = {1-val_split-test_split:.2f}/{val_split:.2f}/{test_split:.2f}")
    print("-" * 55)

    tokenizer = AutoTokenizer.from_pretrained(text_model)
    dataset = MessageDataset(csv_path, tokenizer, max_len=max_len)

    labels = dataset.data['label'].astype(int).values
    idx = np.arange(len(dataset))

    # 3-way stratified split: first carve out TEST, then split TRAIN/VAL.
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_split, random_state=RANDOM_SEED, stratify=labels
    )
    rel_val = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=rel_val, random_state=RANDOM_SEED,
        stratify=labels[trainval_idx]
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Class weighting (pos_weight = #neg / #pos on the TRAIN split)
    tr_labels = labels[train_idx]
    pos = int(tr_labels.sum())
    neg = len(tr_labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float).to(DEVICE)
    print(f"pos_weight (neg/pos): {pos_weight.item():.3f}")

    model = AggressionClassifier(text_model_name=text_model).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    os.makedirs(MODEL_DIR, exist_ok=True)
    tag = 'muril' if 'muril' in text_model.lower() else 'mbert'
    best_path = os.path.join(MODEL_DIR, f'aggression_{tag}.pth')
    best_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            ids = batch['ids'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            tti = batch['token_type_ids'].to(DEVICE)
            y = batch['target'].to(DEVICE)
            logits = model(ids, mask, tti)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if (i + 1) % 200 == 0:
                print(f"  epoch {epoch+1} batch {i+1}/{len(train_loader)} loss {loss.item():.4f}")

        val_loss, vprobs, vlabels = _predict(model, val_loader, criterion, DEVICE)
        acc, p, r, f1, _ = _metrics(vlabels, vprobs)
        print(f"\nEpoch {epoch+1}/{epochs}  train_loss {running/len(train_loader):.4f}")
        print(f"  [VAL] loss {val_loss:.4f} | acc {acc:.4f} | P {p:.4f} | R {r:.4f} | F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ best model saved (val F1={f1:.4f}) -> {best_path}\n")
        else:
            print()

    # --- Final evaluation on the UNTOUCHED test set, using the best checkpoint ---
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    _, tprobs, tlabels = _predict(model, test_loader, criterion, DEVICE)
    acc, p, r, f1, yhat = _metrics(tlabels, tprobs)
    cm = confusion_matrix(tlabels, yhat, labels=[0, 1])
    report = classification_report(
        tlabels, yhat, labels=[0, 1],
        target_names=['Non-Aggressive', 'Aggressive'], zero_division=0
    )

    print("\n" + "=" * 55)
    print(f"STAGE 1 FINAL TEST RESULTS ({text_model})")
    print("=" * 55)
    print(f"Test samples: {len(test_idx)} (held out, model never trained or selected on these)")
    print(f"Accuracy {acc:.4f} | Precision {p:.4f} | Recall {r:.4f} | F1 {f1:.4f}")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(cm)
    print(report)

    # Persist a citable report file
    out_txt = os.path.join(MODEL_DIR, f'eval_stage1_{tag}.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f"Stage 1 (message-level aggression) - {text_model}\n")
        f.write(f"Seed={RANDOM_SEED} | epochs={epochs} | batch={batch_size} | lr={lr}\n")
        f.write(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}\n")
        f.write(f"Model selected on VAL (best F1={best_f1:.4f}); metrics below are on the held-out TEST set.\n\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {p:.4f}\n")
        f.write(f"Recall   : {r:.4f}\n")
        f.write(f"F1       : {f1:.4f}\n\n")
        f.write("Confusion matrix [[TN FP][FN TP]]:\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write(report + "\n")
    print(f"\nSaved test report -> {out_txt}")
    print(f"Best model (selected on val) -> {best_path}")

    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1,
            'best_val_f1': best_f1, 'model_path': best_path, 'report_path': out_txt}


if __name__ == '__main__':
    train_stage1()
