# Pending Work — AI-Based Multilingual Cyberbullying Detection

> **Last Updated:** 2026-04-02  
> **Project Status:** Core codebase complete · Training & evaluation pending

---

## Table of Contents

1. [Overview](#overview)
2. [Priority 1 — Critical (Must Complete)](#priority-1--critical-must-complete)
3. [Priority 2 — Important (Should Complete)](#priority-2--important-should-complete)
4. [Priority 3 — Nice to Have (Optional Enhancements)](#priority-3--nice-to-have-optional-enhancements)
5. [Checklist Summary](#checklist-summary)

---

## Overview

The entire source code architecture is **fully implemented** — model definitions, training loops, evaluation scripts, data pipelines, baseline comparisons, and utility functions are all in place. The remaining work falls into **execution, data preparation, experimentation, and documentation** categories.

### What Is Done

| Component | Status | Notes |
|---|---|---|
| Fusion Model (`fusion_model.py`) | ✅ Complete | m-BERT/MuRIL + ResNet50 + Context MLP with attention fusion |
| Training Pipeline (`train.py`) | ✅ Complete | Train/val split, early stopping, LR scheduler, model checkpointing |
| Evaluation Script (`evaluate.py`) | ✅ Complete | Full metrics computation + confusion matrix |
| Inference Script (`inference.py`) | ✅ Complete | Single-text prediction pipeline |
| Model Comparison (`compare_models.py`) | ✅ Complete | m-BERT vs MuRIL side-by-side evaluation |
| Baseline Models (`baselines.py`) | ✅ Complete | SVM + LSTM implementations |
| Data Preprocessing (`preprocessing.py`) | ✅ Complete | Roman Urdu normalization, text cleaning |
| Dataset Class (`dataset.py`) | ✅ Complete | PyTorch Dataset with image transforms |
| Data Converters (`convert_dataset.py`, `convert_imdb.py`) | ✅ Complete | Excel → CSV and IMDB → CSV converters |
| Metrics Utilities (`metrics.py`) | ✅ Complete | Accuracy, Precision, Recall, F1 |
| Scoring Utilities (`scoring.py`) | ✅ Complete | Repetition & Intent scoring |
| Inter-Annotator Agreement (`fleiss_kappa.py`) | ✅ Complete | Fleiss' Kappa calculation |
| Configuration (`config.py`) | ✅ Complete | Centralized hyperparameters and paths |
| Raw Datasets (`data/`) | ✅ Present | `train.csv`, `train_imdb.csv`, `Roman annotated data.xlsx` |

---

## Priority 1 — Critical (Must Complete)

These tasks are **blockers** — the project cannot be considered functional without them.

### 1.1 Prepare Image Dataset

**Status:** 🔴 Not Started  
**Effort:** Medium  
**Description:**

The multimodal fusion model expects an `images/` directory inside `data/` containing image files referenced by the `image_filename` column in `train.csv`. This directory **does not currently exist**.

**What to do:**

1. Create the directory `data/images/`.
2. Populate it with the actual images referenced in the dataset. Sources may include:
   - Scraped screenshots from social media platforms (Twitter/X, Facebook, Instagram).
   - Publicly available cyberbullying image datasets.
   - Synthetically generated placeholder images if real data is unavailable.
3. Ensure that the filenames in the `image_filename` column of `train.csv` match the actual filenames in `data/images/`.

**Impact if skipped:** The image encoder (ResNet50) will only ever see blank 224×224 images, making the visual modality **completely useless**. The attention-based fusion layer will learn to ignore image features entirely.

**Relevant file:** `src/data/dataset.py` (lines 70–76)

```python
img_name = os.path.join(self.image_dir, str(row['image_filename']))
image = Image.new('RGB', (224, 224))  # Default blank image
try:
    if os.path.exists(img_name):
        image = Image.open(img_name).convert('RGB')
except Exception as e:
    print(f"Error loading image {img_name}: {e}")
```

---

### 1.2 Train the Model

**Status:** 🔴 Not Started  
**Effort:** High (compute-intensive)  
**Description:**

No trained model checkpoints exist. The `models/` directory (where `best_model.pth` and `final_model.pth` are saved) has not been created by the training script.

**What to do:**

1. Ensure all dependencies are installed:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
2. Verify the pipeline works:
   ```bash
   python -m src.test_pipeline
   ```
3. Run training with m-BERT (default):
   ```bash
   python -m src.train
   ```
4. To train with MuRIL instead, update `TEXT_MODEL` in `src/config.py`:
   ```python
   TEXT_MODEL = MURIL_MODEL  # Change from MBERT_MODEL
   ```
   Then re-run training.

**Hardware Considerations:**

| Setup | Estimated Time per Epoch | Notes |
|---|---|---|
| CPU only | 30–60+ minutes | Very slow, not recommended for full training |
| Single GPU (e.g., RTX 3060) | 5–15 minutes | Recommended minimum |
| Cloud GPU (Colab, Kaggle) | 3–10 minutes | Free tier available |

**Output:** Two checkpoint files in `models/`:
- `best_model.pth` — Best validation loss checkpoint
- `final_model.pth` — Last epoch checkpoint

---

### 1.3 Run Full Evaluation

**Status:** 🔴 Not Started (depends on 1.2)  
**Effort:** Low  
**Description:**

After training, generate the final evaluation metrics and confusion matrices.

**What to do:**

```bash
# Evaluate on test set
python -m src.evaluate

# Compare m-BERT vs MuRIL
python -m src.compare_models

# Run baseline comparisons (SVM, LSTM)
python -m src.baselines
```

**Expected Outputs:**
- Per-class Accuracy, Precision, Recall, F1-Score for all 3 labels (Aggression, Repetition, Intent)
- Confusion matrices
- m-BERT vs MuRIL comparison table
- Baseline model (SVM, LSTM) results for comparison

---

### 1.4 Validate Dataset Schema

**Status:** 🟡 Partially Done  
**Effort:** Low  
**Description:**

The `train.csv` file exists but needs verification that it contains all required columns referenced by the codebase.

**Required columns** (referenced in `src/data/dataset.py`):

| Column | Type | Used In |
|---|---|---|
| `text_content` | string | Text preprocessing & tokenization |
| `language` | string (en/ur/roman_ur) | Language-specific preprocessing |
| `image_filename` | string | Image loading from `data/images/` |
| `user_age` | float | Context features |
| `past_flags` | float | Context features |
| `aggression` | int (0/1) | Target label |
| `repetition` | int (0/1) | Target label |
| `intent` | int (0/1) | Target label |

**What to do:**

1. Open `data/train.csv` and verify all 8 columns above are present.
2. Check for missing values / NaN entries.
3. Verify class distribution balance across the 3 labels.
4. Document the dataset statistics (total rows, class distributions, language breakdown).

---

## Priority 2 — Important (Should Complete)

These tasks are essential for a complete FYP submission but do not block model training.

### 2.1 Create EDA Notebooks

**Status:** 🔴 Not Started  
**Effort:** Medium  
**Description:**

The `notebooks/` directory is **completely empty**. For an FYP, Jupyter Notebooks with exploratory data analysis are typically required.

**What to create:**

1. **`notebooks/01_data_exploration.ipynb`** — Dataset overview:
   - Total sample counts per language (Urdu, Roman Urdu, English)
   - Class distribution for Aggression, Repetition, Intent
   - Text length distribution histograms
   - Missing value analysis
   - Sample data visualization

2. **`notebooks/02_preprocessing_demo.ipynb`** — Preprocessing pipeline walkthrough:
   - Raw vs cleaned text examples for each language
   - Roman Urdu normalization examples
   - Tokenization output visualization

3. **`notebooks/03_training_results.ipynb`** — Training analysis (after training):
   - Loss curves (train vs validation)
   - Metric progression across epochs
   - Confusion matrices (heatmaps)
   - Per-class performance breakdown
   - m-BERT vs MuRIL comparison charts
   - Baseline comparison (fusion model vs SVM vs LSTM)

4. **`notebooks/04_inference_demo.ipynb`** — Live prediction demonstrations:
   - Sample predictions across all 3 languages
   - Attention weight visualization (which modality contributed most)
   - Edge case analysis

---

### 2.2 Generate Inter-Annotator Agreement Report

**Status:** 🔴 Not Started  
**Effort:** Low  
**Description:**

The `fleiss_kappa.py` utility is implemented but needs to be executed with actual annotation data.

**What to do:**

```bash
python -m src.utils.fleiss_kappa
```

This validates the reliability of your human annotations. Document the resulting Kappa score and interpretation:

| Kappa Value | Interpretation |
|---|---|
| < 0.20 | Poor agreement |
| 0.21 – 0.40 | Fair agreement |
| 0.41 – 0.60 | Moderate agreement |
| 0.61 – 0.80 | Substantial agreement |
| 0.81 – 1.00 | Almost perfect agreement |

---

### 2.3 Write Final Results Documentation

**Status:** 🔴 Not Started  
**Effort:** Medium  
**Description:**

After all experiments are complete, create a `docs/RESULTS.md` documenting:

1. **Training Configuration** — Final hyperparameters used
2. **Model Performance Table** — Accuracy, Precision, Recall, F1 for each model variant
3. **Comparison Analysis** — m-BERT vs MuRIL strengths/weaknesses
4. **Baseline Comparison** — How the fusion model compares to SVM and LSTM
5. **Ablation Study** (if time permits) — Performance with/without image features, with/without context features
6. **Error Analysis** — Common failure cases and why they occur

---

### 2.4 Set Up Version Control

**Status:** 🔴 Not Started  
**Effort:** Low  
**Description:**

The project does **not have a Git repository** initialized. A `.gitignore` file exists but `git init` has never been run.

**What to do:**

```bash
git init
git add .
git commit -m "Initial commit: complete codebase for cyberbullying detection"
```

Verify that the `.gitignore` properly excludes:
- `venv/`
- `__pycache__/`
- `*.pth` (model checkpoints — too large for Git)
- `data/*.csv` and `data/*.xlsx` (large datasets)

Consider using **Git LFS** or a cloud storage link for large files.

---

## Priority 3 — Nice to Have (Optional Enhancements)

These would strengthen the project but are not strictly required for submission.

### 3.1 Build a Web Demo / API

**Effort:** High  
**Description:**

Create a simple web interface or REST API for live predictions. Options:

- **Streamlit** (fastest): Single-file Python web app
- **Gradio** (easiest sharing): Hugging Face-compatible demo
- **Flask/FastAPI** (most professional): Production-grade API

This would allow demonstrating the model to your FYP panel without running command-line scripts.

---

### 3.2 Add Data Augmentation

**Effort:** Medium  
**Description:**

Improve model robustness with:
- **Text augmentation**: Back-translation, synonym replacement, random insertion/deletion
- **Image augmentation**: Random crop, horizontal flip, color jitter (already partially supported by torchvision transforms)

---

### 3.3 Implement Cross-Validation

**Effort:** Medium  
**Description:**

Replace the single train/val split with k-fold cross-validation (k=5) for more robust performance estimates. This is especially important given the likely small dataset size.

---

### 3.4 Add Logging Framework

**Effort:** Low  
**Description:**

Replace `print()` statements throughout the codebase with Python's `logging` module or integrate **TensorBoard** / **Weights & Biases** for experiment tracking.

```bash
pip install tensorboard
# or
pip install wandb
```

---

### 3.5 Hyperparameter Tuning

**Effort:** High  
**Description:**

Use **Optuna** or **Ray Tune** to systematically search for optimal hyperparameters:
- Learning rate
- Batch size
- Fusion dimension
- Dropout rate
- Scheduler parameters

---

## Checklist Summary

### Critical (Must Do)
- [ ] Create `data/images/` directory and populate with actual images
- [ ] Validate `train.csv` schema has all 8 required columns
- [ ] Run `python -m src.test_pipeline` to verify setup
- [ ] Train model with m-BERT: `python -m src.train`
- [ ] Train model with MuRIL (change config + re-train)
- [ ] Run evaluation: `python -m src.evaluate`
- [ ] Run model comparison: `python -m src.compare_models`
- [ ] Run baselines: `python -m src.baselines`

### Important (Should Do)
- [ ] Create EDA notebook: `notebooks/01_data_exploration.ipynb`
- [ ] Create preprocessing demo: `notebooks/02_preprocessing_demo.ipynb`
- [ ] Create results notebook: `notebooks/03_training_results.ipynb`
- [ ] Create inference demo: `notebooks/04_inference_demo.ipynb`
- [ ] Run inter-annotator agreement: `python -m src.utils.fleiss_kappa`
- [ ] Write `docs/RESULTS.md` with final metrics
- [ ] Initialize Git repository

### Optional (Nice to Have)
- [ ] Build web demo (Streamlit/Gradio)
- [ ] Add data augmentation pipeline
- [ ] Implement k-fold cross-validation
- [ ] Add logging framework (TensorBoard/W&B)
- [ ] Run hyperparameter tuning (Optuna)

---

> **Estimated Total Effort:**  
> - Critical tasks: **2–4 days** (depending on hardware for training)  
> - Important tasks: **3–5 days**  
> - Optional tasks: **5–7 days**  
> - **Full completion: ~2 weeks**
