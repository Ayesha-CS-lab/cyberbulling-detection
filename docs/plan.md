Project Completion Plan — Cyberbullying Detection AI
Current State (What's Actually True)
Item	Reality
Source code	100% complete — no code to write
Git repo	Already initialized (2 commits on main)
Image dataset	All rows use none.jpg — just needs one placeholder file
Trained model	Does not exist — training must be run
Notebooks	notebooks/ is empty
Results docs	No docs/RESULTS.md
Phase 1 — Unblock Training (30 minutes, do first)
Step 1.1 — Create the image placeholder

Every row in train.csv has image_filename = none.jpg. Just create one file:


from PIL import Image
import os
os.makedirs("data/images", exist_ok=True)
Image.new("RGB", (224, 224), color=(128, 128, 128)).save("data/images/none.jpg")
This gives ResNet50 a neutral grey image instead of a runtime error. The visual modality will be uninformative (as expected for a text-dominant dataset), and the attention fusion will learn to down-weight it.

Step 1.2 — Validate dataset schema


python -c "import pandas as pd; df=pd.read_csv('data/train.csv'); print(df.shape, df.columns.tolist(), df.isnull().sum())"
Verify all 8 required columns are present and check for NaN values.

Step 1.3 — Run the pipeline test


python -m src.test_pipeline
This verifies the architecture loads correctly before committing to a long training run.

Phase 2 — Train and Evaluate (2–4 days, compute-dependent)
Step 2.1 — Train with m-BERT (default)


python -m src.train
Produces models/best_model.pth and models/final_model.pth.

Step 2.2 — Train with MuRIL

In src/config.py line 21, change:


TEXT_MODEL = MURIL_MODEL  # was MBERT_MODEL
Then re-run python -m src.train. Save the MuRIL checkpoints separately.

Recommended: Use Google Colab or Kaggle (free GPU). CPU training at 30–60 min/epoch × 10 epochs is painful. The repo can be zipped and uploaded in minutes.

Step 2.3 — Run all evaluations


python -m src.evaluate          # test-set metrics + confusion matrix
python -m src.compare_models    # m-BERT vs MuRIL table
python -m src.baselines         # SVM + LSTM baselines
python -m src.utils.fleiss_kappa  # inter-annotator agreement
Record all output — it feeds Phase 3.

Phase 3 — Create EDA Notebooks (2–3 days)
Four notebooks to create in notebooks/:

Notebook	Content	When
01_data_exploration.ipynb	Sample counts per language, class distributions, text length histograms, missing value analysis	Before training
02_preprocessing_demo.ipynb	Raw vs cleaned text per language, Roman Urdu normalization examples, tokenization output	Before training
03_training_results.ipynb	Loss curves, metric progression, confusion matrix heatmaps, m-BERT vs MuRIL charts, baseline comparison	After training
04_inference_demo.ipynb	Live predictions across 3 languages, attention weight visualization, edge cases	After training
Notebooks 01 and 02 can be done right now (no model needed). Notebooks 03 and 04 need trained weights.

Phase 4 — Documentation (1 day)
Create docs/RESULTS.md with:

Final hyperparameters used
Performance table (Accuracy / Precision / Recall / F1 for each model and each label)
m-BERT vs MuRIL analysis
Fusion model vs SVM vs LSTM comparison
Error analysis section
Phase 5 — Web Demo (Optional but Recommended for FYP Panel)
A Streamlit demo is the fastest option — roughly 100 lines:


pip install streamlit
streamlit run demo.py
The demo would accept text input + language selection and show predictions for Aggression / Repetition / Intent with confidence scores. This is far more impressive to a panel than running a CLI command.

Execution Order Summary

RIGHT NOW (no GPU needed):
  1. Create data/images/none.jpg  ← unblocks everything
  2. Validate train.csv schema
  3. Run src.test_pipeline
  4. Write notebooks/01 and notebooks/02

AFTER TRAINING (GPU recommended):
  5. Train m-BERT model
  6. Train MuRIL model
  7. Run evaluate, compare_models, baselines, fleiss_kappa
  8. Write notebooks/03 and notebooks/04
  9. Write docs/RESULTS.md

OPTIONAL LAST:
  10. Build Streamlit demo





  Step 1 — Upload to Kaggle
Go to kaggle.com → Datasets → New Dataset
Upload kaggle_stage1.zip (Kaggle auto-extracts it)
Name it e.g. cb-stage1, create it
Open a new Notebook, set Accelerator → GPU T4 x2, and Add Input → your cb-stage1 dataset
Step 2 — Setup cell (copy project to writable dir)

import os, shutil, sys

# Find the folder that contains 'src' inside the read-only input
src_root = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'src' in dirs and os.path.exists(os.path.join(root, 'src', 'train_stage1.py')):
        src_root = root; break
print('Found project at:', src_root)

PROJECT = '/kaggle/working/project'
if os.path.exists(PROJECT): shutil.rmtree(PROJECT)
shutil.copytree(src_root, PROJECT)
os.chdir(PROJECT)
if PROJECT not in sys.path: sys.path.insert(0, PROJECT)
print('Working dir:', os.getcwd())
print('messages.csv exists:', os.path.exists('data/processed/messages.csv'))
Step 3 — Train Stage 1 (m-BERT)

!pip install -q transformers
from src.train_stage1 import train_stage1
train_stage1(text_model='bert-base-multilingual-cased', epochs=3, batch_size=16)
Step 4 — Train Stage 1 (MuRIL)

train_stage1(text_model='google/muril-base-cased', epochs=3, batch_size=16)
Step 5 — Download the trained models

import os
print(os.listdir('/kaggle/working/project/models'))
# You'll see aggression_mbert.pth and aggression_muril.pth
Then Save Version (commit), and download via the Kaggle API like before:


kaggle kernels output <your-username>/<notebook-name> -p ./models
Place aggression_mbert.pth and aggression_muril.pth into your local models/ folder.

What happens after Stage 1 is trained
Once you bring those two .pth files back, I'll:

Update demo.py to run the real two-stage pipeline (message → aggression → pair-level cyberbullying decision).
Write docs/RESULTS.md documenting both stages honestly (Stage 2 numbers are already in hand: 91.6% acc, 0.71 F1).