"""
Stage 2 training: user-pair CYBERBULLYING classifier.

Takes the relationship-level features built from the conversation history
(aggression %, repetition, intent-to-harm, peerness, user context) and predicts
the final cyberbullying label CB_Label. This is the "Cyberbullying Assignment
Label" decision layer of the methodology (Figure 1).

A small dense neural network (MLP) is used. Reports:
  - 5-fold stratified cross-validation (mean +/- std)  <- the citable number
  - a final held-out test split (for the saved deployable model)
Both are written to models/eval_stage2.txt for the thesis.

Run:  python -m src.train_stage2
"""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)
import pandas as pd

from src.config import PAIRS_CSV, MODEL_DIR, RANDOM_SEED, DEVICE

NUMERIC_FEATURES = [
    'total_messages', 'aggressive_count', 'aggression_ratio', 'repetition_count',
    'repetition_flag', 'aggression_active_days', 'aggression_span_days',
    'intent_to_harm', 'peerness', 'u1_age', 'u1_grade', 'u2_age', 'u2_grade',
]
GENDER_CATS = ['Male', 'Female', 'Others']


def build_features(df):
    """Return (X, feature_names): numeric features + one-hot gender for both users."""
    parts, names = [], []
    for col in NUMERIC_FEATURES:
        parts.append(df[col].astype(float).values.reshape(-1, 1))
        names.append(col)
    for who in ('u1_gender', 'u2_gender'):
        for cat in GENDER_CATS:
            parts.append((df[who] == cat).astype(float).values.reshape(-1, 1))
            names.append(f'{who}_{cat}')
    X = np.hstack(parts)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # safety guard
    return X, names


class CBClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _standardize(X_tr, X_te):
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    return (X_tr - mean) / std, (X_te - mean) / std, mean, std


def _fit_eval(X_tr, y_tr, X_te, y_te, input_dim, epochs, lr):
    """Train one MLP and return (metrics dict, preds, model, mean, std)."""
    X_tr, X_te, mean, std = _standardize(X_tr, X_te)
    Xt = torch.tensor(X_tr, dtype=torch.float).to(DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float).to(DEVICE)
    Xv = torch.tensor(X_te, dtype=torch.float).to(DEVICE)

    pos = int(y_tr.sum()); neg = len(y_tr) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float).to(DEVICE)

    model = CBClassifier(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        criterion(model(Xt), yt).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xv)).cpu().numpy()
    yhat = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_te, yhat)
    p, r, f1, _ = precision_recall_fscore_support(y_te, yhat, average='binary', zero_division=0)
    return {'acc': acc, 'p': p, 'r': r, 'f1': f1}, yhat, model, mean, std


def train_stage2(csv_path=None, epochs=200, lr=1e-3, test_split=0.2, n_folds=5):
    csv_path = csv_path or PAIRS_CSV
    df = pd.read_csv(csv_path)
    X, feat_names = build_features(df)
    y = df['cb_label'].astype(int).values
    input_dim = X.shape[1]
    print(f"Pairs: {len(df)} | features: {input_dim} | positives: {int(y.sum())}")

    # --- 5-fold stratified cross-validation (the citable number) -------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_rows = []
    print(f"\n=== {n_folds}-fold stratified cross-validation ===")
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        m, _, _, _, _ = _fit_eval(X[tr], y[tr], X[te], y[te], input_dim, epochs, lr)
        fold_rows.append(m)
        print(f"  fold {k}: acc {m['acc']:.4f} | P {m['p']:.4f} | R {m['r']:.4f} | F1 {m['f1']:.4f}")

    def ms(key):
        vals = np.array([fr[key] for fr in fold_rows])
        return vals.mean(), vals.std()

    cv = {k: ms(k) for k in ('acc', 'p', 'r', 'f1')}
    print("\nCV mean +/- std:")
    print(f"  Accuracy  {cv['acc'][0]:.4f} +/- {cv['acc'][1]:.4f}")
    print(f"  Precision {cv['p'][0]:.4f} +/- {cv['p'][1]:.4f}")
    print(f"  Recall    {cv['r'][0]:.4f} +/- {cv['r'][1]:.4f}")
    print(f"  F1        {cv['f1'][0]:.4f} +/- {cv['f1'][1]:.4f}")

    # --- Final held-out test split + the saved deployable model --------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_split, random_state=RANDOM_SEED, stratify=y
    )
    m, yhat, model, mean, std = _fit_eval(X_tr, y_tr, X_te, y_te, input_dim, epochs, lr)
    cm = confusion_matrix(y_te, yhat, labels=[0, 1])
    report = classification_report(
        y_te, yhat, labels=[0, 1],
        target_names=['Not CB', 'Cyberbullying'], zero_division=0
    )
    print("\n=== Final held-out test split ===")
    print(f"Test samples: {len(y_te)}")
    print(f"Accuracy {m['acc']:.4f} | Precision {m['p']:.4f} | Recall {m['r']:.4f} | F1 {m['f1']:.4f}")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(cm)
    print(report)

    # Save model (with scaler stats from the final split)
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt = os.path.join(MODEL_DIR, 'cb_classifier.pth')
    torch.save({
        'state_dict': model.state_dict(),
        'mean': mean, 'std': std,
        'feature_names': feat_names,
        'gender_cats': GENDER_CATS,
        'numeric_features': NUMERIC_FEATURES,
        'input_dim': input_dim,
    }, ckpt)
    print(f"\nSaved Stage 2 model -> {ckpt}")

    # Persist a citable report file
    out_txt = os.path.join(MODEL_DIR, 'eval_stage2.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("Stage 2 (user-pair cyberbullying) - dense MLP\n")
        f.write(f"Seed={RANDOM_SEED} | epochs={epochs} | lr={lr} | features={input_dim}\n")
        f.write(f"Pairs={len(df)} positives={int(y.sum())}\n\n")
        f.write(f"=== {n_folds}-fold stratified cross-validation (mean +/- std) ===\n")
        f.write(f"Accuracy  {cv['acc'][0]:.4f} +/- {cv['acc'][1]:.4f}\n")
        f.write(f"Precision {cv['p'][0]:.4f} +/- {cv['p'][1]:.4f}\n")
        f.write(f"Recall    {cv['r'][0]:.4f} +/- {cv['r'][1]:.4f}\n")
        f.write(f"F1        {cv['f1'][0]:.4f} +/- {cv['f1'][1]:.4f}\n\n")
        for k, fr in enumerate(fold_rows, 1):
            f.write(f"  fold {k}: acc {fr['acc']:.4f} P {fr['p']:.4f} R {fr['r']:.4f} F1 {fr['f1']:.4f}\n")
        f.write(f"\n=== Final held-out test split (n={len(y_te)}) ===\n")
        f.write(f"Accuracy : {m['acc']:.4f}\nPrecision: {m['p']:.4f}\nRecall   : {m['r']:.4f}\nF1       : {m['f1']:.4f}\n\n")
        f.write("Confusion matrix [[TN FP][FN TP]]:\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write(report + "\n")
    print(f"Saved report -> {out_txt}")

    return {'cv': cv, 'test': m}


if __name__ == '__main__':
    train_stage2()
