"""
Stage 3 (image / multimodal) training: a TEXT + IMAGE cyberbullying classifier.

This fulfils the "image-based" / multimodal part of the methodology. Each sample
is a meme (an image plus its overlaid/associated text); the model encodes the
image with ResNet50 and the text with m-BERT/MuRIL, fuses them with an
attention layer (src/models/fusion_model.py), and predicts bullying / not.

It reuses the existing CyberbullyingDetector but with a single binary output.
Context is passed as zeros (meme datasets carry no user context), so the
attention layer simply learns to down-weight it.

Run locally:
    python -m src.train_image --csv data/processed/memes.csv --image-dir data/memes/images

On Kaggle (GPU):
    from src.train_image import train_image
    train_image(csv='data/processed/memes.csv', image_dir='/kaggle/input/.../images',
                text_model='bert-base-multilingual-cased', epochs=5, batch_size=16)
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)
from torchvision import transforms
from transformers import AutoTokenizer

from src.config import DEVICE, MODEL_DIR, MAX_LEN, RANDOM_SEED
from src.data.preprocessing import TextPreprocessor
from src.models.fusion_model import CyberbullyingDetector


def _img_tf(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class MemeDataset(Dataset):
    def __init__(self, df, image_dir, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tok = tokenizer
        self.max_len = max_len
        self.tf = _img_tf()
        self.pre = TextPreprocessor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = self.pre.preprocess(str(row['text_content']), lang=row.get('language', 'en'))
        enc = self.tok(text, add_special_tokens=True, max_length=self.max_len,
                       padding='max_length', truncation=True, return_token_type_ids=True)

        path = os.path.join(self.image_dir, str(row['image_filename']))
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        img = self.tf(img)

        return {
            'ids': torch.tensor(enc['input_ids'], dtype=torch.long),
            'mask': torch.tensor(enc['attention_mask'], dtype=torch.long),
            'tti': torch.tensor(enc.get('token_type_ids', [0] * self.max_len), dtype=torch.long),
            'image': img,
            'context': torch.zeros(2, dtype=torch.float),   # no user context for memes
            'target': torch.tensor(float(row['label']), dtype=torch.float),
        }


def _run(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, preds, ys = 0.0, [], []
    with torch.set_grad_enabled(train):
        for b in loader:
            ids, mask, tti = b['ids'].to(DEVICE), b['mask'].to(DEVICE), b['tti'].to(DEVICE)
            img, ctx, y = b['image'].to(DEVICE), b['context'].to(DEVICE), b['target'].to(DEVICE)
            logits = model(ids, mask, tti, img, ctx).squeeze(-1)   # num_classes=1
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            ys.append(y.cpu().numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    yhat = (preds >= 0.5).astype(int)
    acc = accuracy_score(ys, yhat)
    p, r, f1, _ = precision_recall_fscore_support(ys, yhat, average='binary', zero_division=0)
    return total / len(loader), acc, p, r, f1, ys, yhat


def _freeze(module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def train_image(csv='data/processed/memes.csv', image_dir='data/memes/images',
                text_model='bert-base-multilingual-cased', epochs=15,
                batch_size=16, lr=1e-3, max_len=None,
                freeze_backbones=True, patience=4):
    """
    Multimodal meme classifier. By default the pretrained ResNet50 and m-BERT/MuRIL
    backbones are FROZEN and only the fusion + classifier head is trained — the
    standard transfer-learning recipe for small datasets, which avoids the severe
    overfitting seen when both backbones are fully fine-tuned on a few thousand memes.
    Uses a 70/15/15 train/val/test split, early stopping on validation F1, and
    restores the best epoch before the final test evaluation.
    """
    max_len = max_len or MAX_LEN
    print(f"Device: {DEVICE} | model: {text_model} | csv: {csv} | freeze={freeze_backbones}")
    df = pd.read_csv(csv)
    print(f"Samples: {len(df)} | positives: {int(df['label'].sum())}")

    # 70 / 15 / 15 stratified split
    tr, tmp = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED, stratify=df['label'])
    va_df, te_df = train_test_split(tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=tmp['label'])
    print(f"train {len(tr)} | val {len(va_df)} | test {len(te_df)}")

    tok = AutoTokenizer.from_pretrained(text_model)
    tr_loader = DataLoader(MemeDataset(tr, image_dir, tok, max_len), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(MemeDataset(va_df, image_dir, tok, max_len), batch_size=batch_size)
    te_loader = DataLoader(MemeDataset(te_df, image_dir, tok, max_len), batch_size=batch_size)

    model = CyberbullyingDetector(text_model_name=text_model, num_classes=1).to(DEVICE)
    if freeze_backbones:
        _freeze(model.text_encoder)
        _freeze(model.image_encoder)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"Frozen backbones | trainable params: {sum(p.numel() for p in trainable):,}")
    else:
        trainable = list(model.parameters())

    pos = int(tr['label'].sum()); neg = len(tr) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    import copy
    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        trl, tra, *_ = _run(model, tr_loader, criterion, optimizer)
        vl, va, vp, vr, vf1, *_ = _run(model, va_loader, criterion)
        print(f"epoch {ep+1}/{epochs} | train loss {trl:.4f} acc {tra:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} P {vp:.3f} R {vr:.3f} F1 {vf1:.3f}")
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep+1} (best val F1 {best_f1:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final evaluation on the held-out TEST set
    _, ta, tp, tr_, tf1, ys, yhat = _run(model, te_loader, criterion)
    print("\n=== Stage 3 (multimodal image+text) TEST results (best epoch) ===")
    print(f"Accuracy {ta:.4f} | Precision {tp:.4f} | Recall {tr_:.4f} | F1 {tf1:.4f}")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(ys, yhat))
    print(classification_report(ys, yhat, target_names=['Not Bullying', 'Bullying'], zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    tag = 'mbert' if 'muril' not in text_model else 'muril'
    out = os.path.join(MODEL_DIR, f'image_fusion_{tag}.pth')
    torch.save(model.state_dict(), out)
    print(f"\nSaved -> {out}")
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/processed/memes.csv')
    ap.add_argument('--image-dir', default='data/memes/images')
    ap.add_argument('--text-model', default='bert-base-multilingual-cased')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--no-freeze', action='store_true', help='fine-tune the backbones too (not recommended on small data)')
    args = ap.parse_args()
    train_image(csv=args.csv, image_dir=args.image_dir, text_model=args.text_model,
                epochs=args.epochs, batch_size=args.batch_size,
                freeze_backbones=not args.no_freeze)
