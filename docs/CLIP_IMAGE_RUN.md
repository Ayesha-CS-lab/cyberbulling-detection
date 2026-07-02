# Strong Image Module — CLIP + Hateful Memes (Run Guide)

This rebuilds the image/multimodal component on **CLIP** with the **Hateful Memes**
dataset — the standard strong recipe for harmful-meme detection. The CLIP backbone is
**frozen**; only a small head is trained on its image+text embeddings, which avoids the
overfitting/collapse of the old ResNet50+m-BERT model. It also trains **text-only,
image-only and fusion** heads so you can *prove* multimodality helps.

Code (already written):
- `src/data/convert_hateful_memes.py` — dataset → CSV
- `src/models/clip_model.py` — CLIP classifier + head
- `src/train_clip.py` — feature extraction + train/eval + baselines

**2-day plan:** Day 1 = get data + extract CLIP features (cached) + first results.
Day 2 = tune the head / threshold, finalise the comparison table, write it up.

---

## Step 1 — Add the dataset on Kaggle
New Notebook → **Add Input** → search **"Hateful Memes"** and add a version that
contains a `data/` folder with `img/` and the `.jsonl` files (e.g. *"Facebook Hateful
Meme Dataset"*). Also add your project zip. Accelerator → **GPU T4**.

## Step 2 — Setup
```python
import os, shutil, sys
src_root = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'src' in dirs and os.path.exists(os.path.join(root, 'src', 'train_clip.py')):
        src_root = root; break
PROJECT = '/kaggle/working/project'
if os.path.exists(PROJECT): shutil.rmtree(PROJECT)
shutil.copytree(src_root, PROJECT); os.chdir(PROJECT); sys.path.insert(0, PROJECT)
!pip install -q transformers
```

## Step 3 — Find the data folder, then convert
```python
# locate the folder that holds train.jsonl / dev*.jsonl and img/
for root, dirs, files in os.walk('/kaggle/input'):
    if any(f.endswith('.jsonl') for f in files):
        print('DATA DIR:', root, '| jsonl:', [f for f in files if f.endswith('.jsonl')])
```
```python
DATA = "/kaggle/input/facebook-hateful-meme-dataset/data"   # <- use the path printed above
!python -m src.data.convert_hateful_memes --data-dir "$DATA" --out data/processed/memes_hm.csv
```

## Step 4 — Extract CLIP features + train (this is the whole experiment)
```python
from src.train_clip import run
results = run(csv='data/processed/memes_hm.csv',
              image_dir=DATA,                       # img paths in the CSV are like "img/123.png"
              cache='models/clip_feats.npz')
```
Feature extraction runs once (a few minutes on GPU) and is cached, so re-running to
tune the head is instant. The output prints the table you need:

```
Model                     AUC   macro-F1    Acc
CLIP text-only          0.7x    0.6x       0.6x
CLIP image-only         0.6x    0.5x       0.5x
CLIP fusion (img+txt)   0.7x    0.6x       0.6x   <- should beat both unimodal
```

## Step 5 — Save + record
`Save Version`. Copy the printed table + fusion confusion matrix into
`models/eval_clip.txt`. Bring `models/clip_fusion_head.pth` and `clip_feats.npz` back.

---

## What "strong" looks like (set expectations)
Hateful Memes is hard — published SOTA is ~0.75–0.80 AUC. A frozen-CLIP head landing
around **AUC 0.70–0.75 / macro-F1 ~0.65** is a legitimate, defensible result and a big
jump from the old 0.51. The key thesis story is: **fusion > text-only > image-only**,
shown in the table above.

## If Day-1 results are low
- Try `clip-vit-large-patch14` (bigger CLIP) — change `CLIP_NAME` in `src/train_clip.py`.
- Tune the decision threshold on the val split (report at best-macro-F1 threshold).
- These are quick because features are cached — no re-extraction needed.

## After the run
Send me the results table + confusion matrix and I will update thesis **§3.12** (swap
the architecture description to CLIP) and **§4.7** (replace the preliminary numbers with
the real ones + the fusion-vs-unimodal comparison), then rebuild the document.
