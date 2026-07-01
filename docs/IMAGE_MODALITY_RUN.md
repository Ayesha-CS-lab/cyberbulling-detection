# Image / Multimodal Component — Run Guide (Stage 3)

This adds the **image-based (multimodal)** part of the project promised in the
proposal title. Each sample is a meme (**image + its text**); the model encodes
the image with **ResNet50** and the text with **m-BERT/MuRIL**, fuses them with
an attention layer, and predicts **bullying / not**.

All the code is already written:
- `src/models/fusion_model.py` — ResNet50 + m-BERT attention-fusion network.
- `src/data/convert_memes.py` — converts any meme dataset to the needed CSV.
- `src/train_image.py` — trains and evaluates the multimodal classifier.

You only need to (1) add a dataset and (2) run the notebook on Kaggle GPU.

---

## Step 1 — Add the dataset on Kaggle (only ONE needed)

Open a new Kaggle Notebook → **Add Input** → add:

- **"Memotion Dataset 7k"** (`williamscott701/memotion-dataset-7k`) — ~7k memes with
  OCR text + offensiveness labels. **This single dataset is all you need.**

Also add your project zip (`kaggle_stage1.zip`) as input, exactly as you did for
Stage 1. Set **Accelerator → GPU T4 x2**.

> The `offensive` column has values `not_offensive`, `slight`, `very_offensive`,
> `hateful_offensive`. The converter maps `not_offensive` → 0 (not bullying) and the
> other three → 1 (bullying). The text column used is `text_corrected`.

## Step 2 — Setup cell (copy project to a writable dir)

```python
import os, shutil, sys
src_root = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'src' in dirs and os.path.exists(os.path.join(root, 'src', 'train_image.py')):
        src_root = root; break
PROJECT = '/kaggle/working/project'
if os.path.exists(PROJECT): shutil.rmtree(PROJECT)
shutil.copytree(src_root, PROJECT)
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
print('Working dir:', os.getcwd())
!pip install -q transformers
```

## Step 3 — Confirm the paths, then convert

Memotion 7k usually unpacks to a folder containing `labels.csv` and an `images/`
folder. Confirm the exact paths first (don't assume — Kaggle sometimes nests an
extra folder):

```python
for root, dirs, files in os.walk('/kaggle/input/memotion-dataset-7k'):
    csvs = [f for f in files if f.lower().endswith('.csv')]
    imgs = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if csvs: print('LABELS:', [os.path.join(root, f) for f in csvs])
    if imgs: print('IMAGES dir:', root, '| e.g.', imgs[:2])
```
/kaggle/input/datasets/williamscott701/memotion-dataset-7k
Then run the converter with those two paths (it auto-detects the columns):

```python
!python -m src.data.convert_memes \
    --labels   "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/labels.csv" \
    --image-dir "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images" \
    --out data/processed/memes.csv
```

It should print: `Detected -> image: 'image_name'  text: 'text_corrected'  label:
'offensive'` and the class balance. If a column is wrong, add `--image-col`,
`--text-col`, or `--label-col`.

## Step 4 — Train the multimodal model

Use the **same `--image-dir` path** you confirmed above:

```python
from src.train_image import train_image
IMG = "/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images"
# m-BERT
train_image(csv='data/processed/memes.csv', image_dir=IMG,
            text_model='bert-base-multilingual-cased', epochs=5, batch_size=16)
# (optional) MuRIL
train_image(csv='data/processed/memes.csv', image_dir=IMG,
            text_model='google/muril-base-cased', epochs=5, batch_size=16)
```

The run prints **per-epoch accuracy / precision / recall / F1**, a final
confusion matrix and a classification report — copy these into Chapter 4.

## Step 5 — Save the results

`Save Version` (commit). The trained weights are in
`/kaggle/working/project/models/image_fusion_mbert.pth`. **Copy the printed
metrics text into a file** (e.g. `models/evaluation_report_image.txt`) so the
thesis can cite exact numbers.

---

## If you run out of time

The **text-only thesis is already complete** (`thesis/Cyberbullying_Thesis_Rimsha_Ch.docx`).
The image component is additive: if the Kaggle run does not finish today, submit
the text thesis and add the image chapter later. Nothing here changes or risks
the text work.
