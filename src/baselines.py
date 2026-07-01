"""
Baseline models for the Stage 1 message-level AGGRESSION task.

Provides conventional baselines (SVM + TF-IDF and a BiLSTM) on the SAME
multilingual binary aggression data (data/processed/messages.csv) that the
m-BERT / MuRIL transformer uses, so the comparison in Chapter 4 (Table 4.1,
Section 4.4.2) is like-for-like. Metrics and the split (stratified, seed 42)
match src/train_stage1.py.

Run:  python -m src.baselines            # SVM (fast)
      python -m src.baselines --lstm     # also run the BiLSTM (slow on CPU)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)

from src.config import MESSAGES_CSV, RANDOM_SEED, MODEL_DIR
from src.data.preprocessing import TextPreprocessor

TEST_SPLIT = 0.15


def _load():
    df = pd.read_csv(MESSAGES_CSV)
    pre = TextPreprocessor()
    texts = [pre.preprocess(str(t), lang=l) for t, l in zip(df['message'], df['language'])]
    y = df['label'].astype(int).values
    return np.array(texts, dtype=object), y


def _report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"\n--- {name} ---")
    print(f"Accuracy {acc:.4f} | Precision {p:.4f} | Recall {r:.4f} | F1 {f1:.4f}")
    print("Confusion [[TN FP][FN TP]]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(classification_report(y_true, y_pred, labels=[0, 1],
          target_names=['Non-Aggressive', 'Aggressive'], zero_division=0))
    return {'name': name, 'acc': acc, 'p': p, 'r': r, 'f1': f1}


def svm_baseline(texts, y):
    Xtr, Xte, ytr, yte = train_test_split(
        texts, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    Xtr_v = tfidf.fit_transform(Xtr)
    Xte_v = tfidf.transform(Xte)
    # class_weight balanced = the SVM analogue of Stage 1's pos_weight
    clf = LinearSVC(class_weight='balanced', random_state=RANDOM_SEED)
    clf.fit(Xtr_v, ytr)
    return _report("SVM + TF-IDF", yte, clf.predict(Xte_v))


def lstm_baseline(texts, y, epochs=5):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from collections import Counter

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[LSTM] device: {device}")

    wf = Counter()
    for t in texts:
        wf.update(t.lower().split())
    vocab = {w: i + 2 for i, (w, _) in enumerate(wf.most_common(20000))}
    vocab['<PAD>'] = 0; vocab['<UNK>'] = 1
    max_len = 64

    def enc(t):
        toks = t.lower().split()[:max_len]
        ids = [vocab.get(w, 1) for w in toks]
        return ids + [0] * (max_len - len(ids))

    X = np.array([enc(t) for t in texts])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)

    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float)),
                        batch_size=64, shuffle=True)

    class BiLSTM(nn.Module):
        def __init__(self, vsize, emb=128, hid=128):
            super().__init__()
            self.emb = nn.Embedding(vsize, emb, padding_idx=0)
            self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(hid * 2, 1)

        def forward(self, x):
            e = self.emb(x)
            _, (h, _) = self.lstm(e)
            h = self.drop(torch.cat([h[-2], h[-1]], dim=1))
            return self.fc(h).squeeze(-1)

    model = BiLSTM(len(vocab) + 2).to(device)
    pos = int(ytr.sum()); neg = len(ytr) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(epochs):
        model.train(); tot = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[LSTM] epoch {ep+1}/{epochs} loss {tot/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte).to(device))
        pred = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)
    return _report("BiLSTM", yte, pred)


def main(run_lstm=False):
    texts, y = _load()
    print(f"Messages: {len(y)} | aggressive: {int(y.sum())}")
    results = [svm_baseline(texts, y)]
    if run_lstm:
        results.append(lstm_baseline(texts, y))

    print("\n" + "=" * 56)
    print("STAGE 1 BASELINE COMPARISON (aggression task)")
    print("=" * 56)
    print(f"{'Model':<16}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
    for r in results:
        print(f"{r['name']:<16}{r['acc']:>8.4f}{r['p']:>8.4f}{r['r']:>8.4f}{r['f1']:>8.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'eval_baselines.txt'), 'w', encoding='utf-8') as f:
        f.write("Stage 1 baselines on messages.csv (binary aggression)\n")
        f.write(f"Seed={RANDOM_SEED}, test_split={TEST_SPLIT}\n\n")
        for r in results:
            f.write(f"{r['name']}: acc {r['acc']:.4f} P {r['p']:.4f} R {r['r']:.4f} F1 {r['f1']:.4f}\n")
    print("\nSaved -> models/eval_baselines.txt")


if __name__ == '__main__':
    main(run_lstm='--lstm' in sys.argv)