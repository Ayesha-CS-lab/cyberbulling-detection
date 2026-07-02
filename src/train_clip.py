"""
Train the CLIP-based multimodal meme classifier (Stage 3, rebuilt).

Pipeline:
  1. Extract FROZEN CLIP image + text embeddings for every meme (cached to .npy).
  2. Train a small MLP head on those features for three modes — text-only,
     image-only and fusion — so the benefit of multimodality is measurable.
  3. Select the best epoch on a validation split by ROC-AUC (never single-class F1,
     which caused the earlier majority-class collapse) and report on the held-out
     dev set with AUC, macro-F1, accuracy and a confusion matrix.

Run on Kaggle (GPU) after converting the dataset:
    from src.train_clip import run
    run(csv='data/processed/memes_hm.csv',
        image_dir='/kaggle/input/hateful-memes/data')
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix,
                             classification_report)

from src.config import DEVICE, MODEL_DIR, RANDOM_SEED
from src.models.clip_model import HeadOnly

CLIP_NAME = 'openai/clip-vit-base-patch32'


def _pool(out):
    """Pooled representation, robust across transformers versions."""
    p = getattr(out, 'pooler_output', None)
    return p if p is not None else out.last_hidden_state[:, 0]


# ── 1. Feature extraction (frozen CLIP) ──────────────────────────────────────
def extract_features(df, image_dir, cache=None, batch_size=64):
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"Loaded cached features from {cache}")
        return d['img'], d['txt'], d['y'], d['split']

    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    img_embs, txt_embs = [], []
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i + batch_size]
        images = []
        for f in chunk['image_filename']:
            try:
                images.append(Image.open(os.path.join(image_dir, str(f))).convert('RGB'))
            except Exception:
                images.append(Image.new('RGB', (224, 224)))
        inputs = proc(text=list(chunk['text_content'].astype(str)), images=images,
                      return_tensors='pt', padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            vout = clip.vision_model(pixel_values=inputs['pixel_values'].to(DEVICE))
            tout = clip.text_model(input_ids=inputs['input_ids'].to(DEVICE),
                                   attention_mask=inputs['attention_mask'].to(DEVICE))
            im = clip.visual_projection(_pool(vout))   # raw 768 -> 512
            tx = clip.text_projection(_pool(tout))     # raw 512 -> 512
        im = im / im.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        tx = tx / tx.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        img_embs.append(im.cpu().numpy())
        txt_embs.append(tx.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  features {i+len(chunk)}/{len(df)}")

    img = np.concatenate(img_embs).astype(np.float32)
    txt = np.concatenate(txt_embs).astype(np.float32)
    y = df['label'].astype(int).values
    split = df['split'].values
    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        np.savez_compressed(cache, img=img, txt=txt, y=y, split=split)
        print(f"Cached features -> {cache}")
    return img, txt, y, split


# ── 2. Train one head (text / image / fusion) ────────────────────────────────
def _features_for_mode(img, txt, mode):
    if mode == 'fusion':
        return np.concatenate([img, txt], axis=1)
    return img if mode == 'image' else txt


def train_head(X_tr, y_tr, X_va, y_va, X_te, y_te, epochs=60, lr=1e-3):
    Xt = torch.tensor(X_tr).to(DEVICE); yt = torch.tensor(y_tr, dtype=torch.float).to(DEVICE)
    Xv = torch.tensor(X_va).to(DEVICE)
    Xe = torch.tensor(X_te).to(DEVICE)
    pos = int(y_tr.sum()); neg = len(y_tr) - pos
    pw = torch.tensor([neg / max(pos, 1)], dtype=torch.float).to(DEVICE)
    model = HeadOnly(X_tr.shape[1]).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Xt), yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)).cpu().numpy()
        try:
            auc = roc_auc_score(y_va, pv)
        except ValueError:
            auc = 0.0
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pe = torch.sigmoid(model(Xe)).cpu().numpy()
    yhat = (pe >= 0.5).astype(int)
    return {
        'auc': roc_auc_score(y_te, pe),
        'macro_f1': f1_score(y_te, yhat, average='macro'),
        'acc': accuracy_score(y_te, yhat),
        'cm': confusion_matrix(y_te, yhat),
        'yhat': yhat, 'probs': pe, 'model': model,
    }


# ── 3. Orchestration ─────────────────────────────────────────────────────────
def run(csv='data/processed/memes_hm.csv', image_dir='data/hateful_memes',
        cache='models/clip_feats.npz'):
    df = pd.read_csv(csv)
    print(f"Samples: {len(df)} | positives: {int(df['label'].sum())}")
    img, txt, y, split = extract_features(df, image_dir, cache=cache)

    is_dev = (split == 'dev')
    img_tr_all, txt_tr_all, y_tr_all = img[~is_dev], txt[~is_dev], y[~is_dev]
    img_te, txt_te, y_te = img[is_dev], txt[is_dev], y[is_dev]
    # carve a validation split from train for model selection
    idx = np.arange(len(y_tr_all))
    tr_i, va_i = train_test_split(idx, test_size=0.15, random_state=RANDOM_SEED, stratify=y_tr_all)

    print(f"train {len(tr_i)} | val {len(va_i)} | test(dev) {len(y_te)}")
    results = {}
    for mode in ['text', 'image', 'fusion']:
        Xtr = _features_for_mode(img_tr_all, txt_tr_all, mode)
        Xte = _features_for_mode(img_te, txt_te, mode)
        r = train_head(Xtr[tr_i], y_tr_all[tr_i], Xtr[va_i], y_tr_all[va_i], Xte, y_te)
        results[mode] = r
        print(f"  {mode:6} -> AUC {r['auc']:.4f} | macro-F1 {r['macro_f1']:.4f} | acc {r['acc']:.4f}")

    print("\n================ RESULTS (held-out dev) ================")
    print(f"{'Model':22} {'AUC':>7} {'macro-F1':>9} {'Acc':>7}")
    names = {'text': 'CLIP text-only', 'image': 'CLIP image-only', 'fusion': 'CLIP fusion (img+txt)'}
    for m in ['text', 'image', 'fusion']:
        r = results[m]
        print(f"{names[m]:22} {r['auc']:7.4f} {r['macro_f1']:9.4f} {r['acc']:7.4f}")
    print("\nFusion confusion matrix [[TN FP][FN TP]]:")
    print(results['fusion']['cm'])
    print(classification_report(y_te, results['fusion']['yhat'],
                                target_names=['Not hateful', 'Hateful'], zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, 'clip_fusion_head.pth')
    torch.save({'state_dict': results['fusion']['model'].state_dict(),
                'clip_name': CLIP_NAME, 'mode': 'fusion', 'in_dim': img.shape[1] * 2}, out)
    print(f"\nSaved fusion head -> {out}")
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/processed/memes_hm.csv')
    ap.add_argument('--image-dir', default='data/hateful_memes')
    ap.add_argument('--cache', default='models/clip_feats.npz')
    args = ap.parse_args()
    run(csv=args.csv, image_dir=args.image_dir, cache=args.cache)
