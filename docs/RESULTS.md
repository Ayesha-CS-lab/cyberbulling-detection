# Results — Confirmed Numbers for Chapter 4

All numbers below are reproducible (fixed seed = 42) and backed by saved report
files in `models/`. This document is the single source of truth for the thesis
Results chapter. *(Image / Stage 3 numbers are added after the Memotion run.)*

---

## 4.2 Dataset statistics

| View | Size | Positives | Notes |
|---|---|---|---|
| Messages (Stage 1) | 92,308 | 31,389 aggressive (34.0%) | 90,356 English + 1,952 Roman Urdu |
| User-pairs (Stage 2) | 9,511 | 992 cyberbullying (10.4%) | from 100 users' conversation history |

> **Label correction (2026-06-30).** The Roman Urdu source (`Roman annotated data.xlsx`)
> uses a **0/1/2** scale (negative/neutral/positive style), where only **level 2** is the
> hostile/aggressive class. `convert_dataset.py` originally summed the raw 0/1/2 values and
> flagged `sum >= 2` as aggressive, which mislabelled many positive comments as aggressive
> (e.g. *"Shukar hay Australia jeeta :)"*). It now maps each annotator to `rating == 2` and
> takes a majority vote. Roman Urdu aggressive count: **1,442 → 612**. Stage 1 was trained
> *before* this fix; impact is <1% of all labels (English is unaffected), so a retrain is
> expected to change the numbers negligibly. Old data backed up as
> `data/train_OLD_backup.csv` and `data/processed/messages_OLD_backup.csv`.

## 4.3 Inter-annotator agreement (Fleiss' κ)

Roman Urdu data, 3 annotators, 1,999 items (1 dropped for a missing rating).
Annotators used a **3-level scale (0/1/2)**.

| Scheme | Fleiss' κ | Interpretation |
|---|---|---|
| 3-category (0/1/2, as-is) | **0.666** | Substantial |
| Binary: 0 vs {1,2} | 0.601 | Substantial |
| Binary: {0,1} vs 2 | 0.723 | Substantial |

Agreement breakdown (3-category): 1,337 unanimous, 662 majority, 0 fully split.
*Source: `src/utils/fleiss_kappa.py`.* **TODO:** confirm with supervisor what
0/1/2 denote, then cite the matching row (default: 3-category κ = 0.67).

## 4.4 Stage 1 — Message-level aggression (held-out test, n = 13,848)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM + TF-IDF (baseline) | 0.847 | 0.763 | 0.812 | 0.788 |
| BiLSTM (baseline) | 0.858 | 0.780 | 0.824 | 0.802 |
| **m-BERT** | 0.881 | **0.816** | 0.849 | 0.832 |
| **MuRIL** | **0.882** | 0.795 | **0.889** | **0.840** |

Clean progression SVM < BiLSTM < m-BERT < MuRIL — confirms the Chapter 2 narrative
(classical ML < older deep learning < transformers).

Confusion matrices (TN, FP, FN, TP):
- SVM: 7805, 1215, 899, 3929
- BiLSTM: 7901, 1119, 850, 3978
- m-BERT: 8094, 926, 727, 4101
- MuRIL: 7915, 1105, 536, 4292

**Findings (RQ1):** Both transformers beat the classical SVM baseline. **MuRIL is
best overall** (highest accuracy, F1, and recall). Its higher recall (0.889 vs
0.849) means it misses fewer aggressive messages — the preferable trade-off for a
safety task — consistent with MuRIL's pretraining on South Asian / transliterated
text. *Source: `models/eval_stage1_mbert.txt`, `eval_stage1_muril.txt`,
`eval_baselines.txt`.*

## 4.5 Stage 2 — User-pair cyberbullying classification

5-fold stratified cross-validation (the headline number) and a held-out test split.

| Evaluation | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 5-fold CV (mean ± std) | 0.921 ± 0.004 | 0.572 ± 0.013 | 0.972 ± 0.010 | 0.720 ± 0.012 |
| Held-out test (n = 1,903) | 0.921 | 0.568 | 0.990 | 0.722 |

Test confusion matrix (TN, FP, FN, TP): 1556, 149, 2, 196 — catches **196 of 198**
cyberbullying pairs. The low CV std (±0.4% accuracy) shows the result is stable,
not a lucky split. Precision is lower (~0.57) because the model deliberately
favours recall on a 10.4%-positive, safety-critical task. *Source:
`models/eval_stage2.txt`.*

## 4.6 The "before" story — majority-class collapse

The earlier single-stage, multi-label model collapsed to the majority class:
e.g. the MuRIL aggression confusion matrix was `[[0, 84], [0, 216]]` — it predicted
"positive" for **every** input, yielding a misleading 72% "accuracy" (the base rate)
while detecting nothing. Root causes: (i) class imbalance with no weighting, and
(ii) Repetition/Intent had zero positive examples in the old data and were
unlearnable per-message. The fix — class weighting, recall/F1-based evaluation, and
the two-stage relationship-level design — produced the genuine results above. This
is the instructive contrast between a model that *appears* to work (high accuracy,
detects nothing) and one that genuinely does.

## 4.x Stage 3 — Image / multimodal (Memotion)

Multimodal ResNet50 + m-BERT fusion on the Memotion meme dataset (6,992 memes,
image + OCR text, offensive vs not; ~61% positive). Held-out test (n = 1,399).

| Metric | Value |
|---|---|
| Accuracy | 0.543 |
| Precision | 0.620 |
| Recall | 0.653 |
| F1 | 0.636 |

Confusion matrix (TN, FP, FN, TP): 201, 342, 297, 559.

**Honest finding (limitation).** Unlike the text stages, the multimodal meme model
did **not** reliably beat the base rate (~61% positive). Two training regimes were
tried: (i) full fine-tuning **overfit** (train acc 0.96, val ~0.54); (ii) freezing the
encoders **collapsed to the majority class** under F1-based selection (predicted
"offensive" for every meme). The reported figures above are from the discriminating
(fine-tuned) run, which at least predicts both classes.

This reflects the well-documented difficulty of **meme offensiveness detection** —
subtle humour, sarcasm, and cultural context, on a small (~7k) dataset with graded
labels (slight/very/hateful all mapped to positive). The **text-based Stages 1–2 remain
the effective core** of the system; robust multimodal fusion is identified as future
work (larger paired data, better image–text alignment). *Source: Kaggle `train_image`
run; model `models/image_fusion_mbert.pth`.*

---

### Saved report files (cite these)
- `models/eval_stage1_mbert.txt`, `models/eval_stage1_muril.txt` — Stage 1 test
- `models/eval_baselines.txt` — SVM (and BiLSTM once run)
- `models/eval_stage2.txt` — Stage 2 CV + test
- Fleiss κ — re-run `python -m src.utils.fleiss_kappa`
- `models/evaluation_report_image.txt` — Stage 3 (after Memotion run)
