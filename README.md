# AI-Based Multilingual Cyberbullying Detection

A deep learning framework for detecting cyberbullying across Urdu, Roman Urdu, and English text using m-BERT, MuRIL, and ResNet50.

## Project Structure

```
src/
├── config.py              # Centralized hyperparameters
├── train.py               # Training with validation & early stopping
├── evaluate.py            # Full evaluation with metrics & confusion matrix
├── inference.py           # Run predictions on new text
├── compare_models.py      # m-BERT vs MuRIL comparison
├── baselines.py           # SVM + LSTM baselines
├── test_pipeline.py       # Model architecture verification
├── data/
│   ├── preprocessing.py   # Text cleaning & Roman Urdu normalization
│   ├── dataset.py         # PyTorch Dataset with image transforms
│   ├── convert_dataset.py # Convert Roman Urdu Excel → CSV
│   └── convert_imdb.py    # Convert IMDB dataset → CSV
├── models/
│   └── fusion_model.py    # Multimodal fusion (Text+Image+Context)
└── utils/
    ├── metrics.py         # Accuracy, Precision, Recall, F1
    ├── scoring.py         # Repetition & Intent scoring
    └── fleiss_kappa.py    # Inter-annotator agreement
```

## Setup

1. **Create Virtual Environment:**

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies:**

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

3. **Verify Installation:**
   ```bash
   python -m src.test_pipeline
   ```

## Data Processing

1. **Convert Roman Urdu dataset:**

   ```bash
   python src/data/convert_dataset.py
   ```

2. **Convert IMDB dataset:**
   ```bash
   python src/data/convert_imdb.py
   ```

## Usage

### Train the Model

```bash
python -m src.train
```

### Evaluate on Test Set

```bash
python -m src.evaluate
```

### Run Predictions

```bash
python -m src.inference
```

### Compare m-BERT vs MuRIL

```bash
python -m src.compare_models
```

### Run Baseline Models (SVM, LSTM)

```bash
python -m src.baselines
```

### Calculate Inter-Annotator Agreement

```bash
python -m src.utils.fleiss_kappa
```

### Test Repetition & Intent Scoring

```bash
python -m src.utils.scoring
```

## Model Details

- **Text Encoder:** `bert-base-multilingual-cased` (m-BERT) or `google/muril-base-cased` (MuRIL)
- **Image Encoder:** Pre-trained ResNet50
- **Fusion:** Attention-based mechanism combining text, image, and context features
- **Output:** 3 labels — Aggression, Repetition, Intent to Harm
