# CHAPTER 3

# MATERIALS AND METHODS

## 3.1 Overview of the Proposed Methodology

The idea that drives this whole design is simple: **cyberbullying lives in a relationship over
time, not in a single message**. Judge one message on its own and the best you can do is spot
*aggression* — you cannot see the *repetition* and *intent* that separate real bullying from a
one-off insult (Sections 1.4 and 2.2). So the system is split into **two stages** (Figure 3.1):

- **Stage 1 — Message-level aggression.** A multilingual transformer (m-BERT or MuRIL) is
  fine-tuned to say whether one message — English or Roman Urdu — is aggressive. This is the
  language building block.
- **Stage 2 — Relationship-level cyberbullying.** For each pair of users, the aggressive-message
  signal is gathered across their whole conversation and combined with measures of
  **repetition**, **intent to harm**, **peerness** and **user context** (age, grade, gender). A
  small dense network then makes the final call for that relationship.

The split lines up with the behavioural definition from Chapter 2: Stage 1 gives the
*aggression* pillar at the message level, and Stage 2 rebuilds *repetition*, *intent* and
*peerness* at the relationship level. The pipeline runs in five steps — data collection and
preprocessing, annotation, feature extraction, model development (Stages 1 and 2), and
evaluation — each covered below.

> **Figure 3.1.** Overall architecture of the two-stage system. Stage 1 (m-BERT / MuRIL) marks
> individual messages as aggressive; Stage 2 aggregates these over each pair's history and adds
> repetition, intent, peerness and context to produce the final cyberbullying label.

## 3.2 Dataset

### 3.2.1 The Comprehensive Cyberbullying Dataset

The main data comes from the comprehensive cyberbullying dataset of **Ejaz, Razi and Choudhury
(2024)** [3]. What sets it apart from most public sets is that it labels not just aggression but
also the **repetition, peerness and intent to harm** that make up cyberbullying. It is stored as
a set of linked tables describing a community of users and their messages:

- **Users** — age, gender, school and grade for each user.
- **Peerness values** — a number describing the social relationship between two users, based on
  their relative age and grade.
- **Communication data** — the message log between users; every message has an aggression label
  and a timestamp.
- **CB labels** — the ground-truth cyberbullying label for each user-pair, plus aggregate counts
  (total messages, aggressive count, intent to harm, peerness).

Because each message is time-stamped and tied to a sender and receiver, the data supports exactly
the **temporal, relationship-level** analysis that the definition of cyberbullying calls for.

### 3.2.2 Roman Urdu Aggression Data

To make Stage 1 genuinely **multilingual**, the English communication data is combined with a
**Roman Urdu** set of annotated aggressive and non-aggressive comments. Each Roman Urdu record
adds its text and a binary aggression label, so the transformer learns aggression cues in both
languages inside one shared model. The reliability of the Roman Urdu labels is checked separately
with Fleiss' Kappa (Section 3.4.2).

### 3.2.3 Message-Level and Pair-Level Views

A preparation script (`prepare_dataset.py`) turns these raw tables into the two training tables
the system uses (Figure 3.2):

- **`messages.csv` (Stage 1 view).** One row per message — *message*, *label* (aggressive /
  non-aggressive), *language*. The English log and the Roman Urdu set are joined, blank messages
  dropped and exact duplicates removed. The result is **92,308 messages** — **90,356 English**
  and **1,952 Roman Urdu** — of which **31,389 are aggressive** and **60,919 non-aggressive**
  (Table 3.1).
- **`pairs.csv` (Stage 2 view).** One row per user-pair, with eighteen columns holding the
  aggregated behavioural and contextual features plus the final label. There are **9,511
  user-pairs**, and only **992 (about 10.4%)** are cyberbullying against **8,519** that are not —
  a heavy imbalance that shapes the design and evaluation of Stage 2 (Sections 3.8 and 3.9).

> **Figure 3.2.** Building the message-level and pair-level views from the raw dataset tables.

**Table 3.1.** Composition of the message-level dataset by language.

| Language | Messages | Share |
|---|---:|---:|
| English | 90,356 | 97.9% |
| Roman Urdu | 1,952 | 2.1% |
| **Total** | **92,308** | **100%** |
| *of which aggressive* | *31,389* | *34.0%* |
| *of which non-aggressive* | *60,919* | *66.0%* |

To make the data concrete, **Table 3.2** shows a few real annotated messages from both languages.
The Roman Urdu rows show the code-switched, transliterated style discussed earlier, with English
glosses in brackets. (Strong profanity is masked here.)

**Table 3.2.** Sample annotated messages from the dataset.

| Language | Message (English gloss) | Label |
|---|---|---|
| English | "The above user is a nutter and has been banned from Wikipedia" | Aggressive |
| English | "U c\*\*\* why did u block me" | Aggressive |
| English | "In fact it was just speedied as a hoax" | Non-aggressive |
| Roman Urdu | "Roni soraton :p" (*cry-faces* — mocking the losers) | Aggressive |
| Roman Urdu | "Tabhi kehtay hain achay bachay bet nahi lagatay, ab bhukto" (*that's why they say good kids don't bet — now suffer*) | Aggressive |
| Roman Urdu | "Shukar hay Australia jeeta :)" (*thank God Australia won*) | Non-aggressive |
| Roman Urdu | "mjhy pta tha k aus hi jeety ga :p" (*I knew Australia would win*) | Non-aggressive |

### 3.3 Data Preprocessing

#### 3.3.1 Text Cleaning and Normalisation

Every message goes through a cleaning step (`TextPreprocessor`) before tokenisation. The step
strips out things that carry no aggression signal and would only add noise:

- **URLs** (`http(s)://…`, `www.…`) are removed.
- **HTML tags** are removed.
- **User mentions** (`@username`) are removed — who is mentioned says nothing about whether the
  message is aggressive.
- **Emojis and stray symbols** are replaced with spaces.
- **Whitespace** is collapsed so runs of spaces become one.

#### 3.3.2 Roman Urdu Handling and Tokenisation

Roman Urdu is tricky because it has **no standard spelling** — one word can be written many ways.
A light normalisation step lower-cases the text and folds a few common repeated-vowel spellings
(for example collapsing doubled vowels), so near-identical spellings move closer together. This
is kept deliberately gentle; the real work of generalising across languages and spellings is left
to the transformer, whose sub-word tokeniser and shared representation handle spelling variation
far better than any fixed rule set could. Each cleaned message is then tokenised with the model's
own pretrained tokeniser (WordPiece for m-BERT, the MuRIL tokeniser for MuRIL) and padded or
truncated to a fixed length.

## 3.4 Data Annotation and Inter-Annotator Agreement

### 3.4.1 Annotation Scheme

The data is labelled against the four pillars from Chapter 1:

- **Aggression** — a binary label on each message: hostile/abusive or not.
- **Repetition** — taken from each pair's history: how many aggressive acts there are and how
  persistently they come back over time.
- **Intent to harm** — how strongly the language aims to threaten, degrade, intimidate or isolate.
- **Peerness** — the relationship and power balance between the two users, from their relative age
  and grade.

The per-relationship **cyberbullying label (CB_Label)** is the ground truth Stage 2 learns to
predict.

The Roman Urdu part came with a **three-level (0/1/2)** scale — neutral, positive and
hostile/negative tone — rated independently by three annotators. For the aggression task, only the
**hostile (level 2)** class counts as aggressive: each annotator's rating becomes a binary
"aggressive" vote (`rating == 2`), and a **majority vote** (at least two of three) sets the final
label. This gives 612 aggressive Roman Urdu messages, which feed the Stage 1 labels.

### 3.4.2 Fleiss' Kappa Agreement

Since labelling abuse involves judgement calls, the **reliability** of the labels has to be shown,
not assumed. Each Roman Urdu item was rated by **three annotators**, and their agreement is
measured with **Fleiss' Kappa** (`fleiss_kappa.py`) — the standard chance-corrected agreement
score for several raters on categorical labels. For two categories and three raters, a ratings
matrix records how many annotators chose each category per item, and Kappa is

  κ = (P̄ − Pₑ) / (1 − Pₑ),

where P̄ is the mean observed agreement and Pₑ is the agreement expected by chance. The result is
read on the usual scale (slight, fair, moderate, substantial, almost-perfect), and the share of
items with unanimous versus majority agreement is reported with it (Section 4.3). Reporting Kappa
is what shows the labels behind the model are consistent.

## 3.5 Feature Extraction

### 3.5.1 Textual Features (m-BERT / MuRIL Embeddings)

Meaning in the text is captured by a **multilingual transformer**, and two are compared:

- **m-BERT (`bert-base-multilingual-cased`)** — pretrained on Wikipedia in over a hundred
  languages, giving a shared multilingual representation that suits code-switched English/Roman
  Urdu.
- **MuRIL (`google/muril-base-cased`)** — pretrained specifically on South Asian languages and on
  transliterated text, which is directly relevant to Roman Urdu.

Each produces a 768-dimensional contextual embedding per message. Unlike bag-of-words or lexicon
features, these embeddings carry word order, context and cross-lingual links, so the model reads
meaning that depends on *how* words are used, not just *which* words appear.

### 3.5.2 Behavioural and Contextual Features

At the relationship level, each user-pair is summarised by a vector of **behavioural and
contextual features** built from its history (Table 3.3): how much the two talk, how much of it is
aggressive, temporal repetition signals (how many distinct days aggression happened on, and over
how long), the intent-to-harm score, the peerness score, and the demographics of both users.
Together these turn a whole relationship into one fixed-length vector that Stage 2 can classify.

**Table 3.3.** Pair-level relationship features used by the Stage 2 classifier.

| Feature | Description |
|---|---|
| `total_messages` | Total messages exchanged between the two users |
| `aggressive_count` | Number of aggressive messages in the relationship |
| `aggression_ratio` | Aggressive messages ÷ total messages |
| `repetition_count` | Count of aggressive acts (repetition signal) |
| `repetition_flag` | 1 if there are ≥ 2 aggressive messages, else 0 |
| `aggression_active_days` | Number of distinct days on which aggression occurred |
| `aggression_span_days` | Days between the first and last aggressive message |
| `intent_to_harm` | Intent-to-harm score for the relationship |
| `peerness` | Social relationship / power balance between the two users |
| `u1_age`, `u1_grade`, `u1_gender` | Demographic context of user 1 |
| `u2_age`, `u2_grade`, `u2_gender` | Demographic context of user 2 |

## 3.6 Stage 1 — Message-Level Aggression Model

### 3.6.1 Transformer Fine-Tuning

Stage 1 (`AggressionClassifier`) fine-tunes the chosen transformer for binary aggression
classification (Figure 3.4). A message is tokenised and run through the m-BERT/MuRIL encoder; the
pooled `[CLS]` vector (768 dimensions) is taken as the message embedding, passed through a dropout
layer, and mapped by one linear layer to a single logit. A sigmoid turns that into the probability
the message is aggressive. The whole network — encoder and head — is fine-tuned end-to-end with
binary cross-entropy.

> **Figure 3.4.** Stage 1 architecture: a multilingual transformer encoder (m-BERT / MuRIL) with
> dropout and a single linear head producing an aggression probability.

### 3.6.2 Class Imbalance Handling

Aggressive messages are the minority (about 35%; Table 3.1). Left alone, that pushes a model to
favour the non-aggressive class. To stop this, the loss uses **class weighting** so mistakes on the
aggressive class hurt more, and training is watched with **precision, recall and F1** rather than
accuracy alone. This choice is a direct answer to the majority-class collapse seen in an earlier
version (Sections 1.6 and 4.6).

## 3.7 Quantifying Repetition and Intent

One of the contributions here is that repetition and intent are **measured**, not assumed. Two
scoring components (`scoring.py`) do this, and the temporal repetition features used by Stage 2 are
computed directly from the timestamped log.

### 3.7.1 Repetition Scoring

Repetition means **the same aggression coming back between the same two users over time**. Two
signals capture it. From the log, the pipeline counts how many **distinct days** aggression
occurred on (`aggression_active_days`) and the **span** between the first and last aggressive
message (`aggression_span_days`), and a `repetition_flag` marks pairs with two or more aggressive
messages — **5,568 of the 9,511 pairs** trip that flag. On the content side, a `RepetitionScorer`
compares a user's messages with **Jaccard similarity** over their word sets, counts near-duplicate
pairs above a threshold, and weights the result by frequency, so a stream of repeated attacks
scores high while a single message scores zero (Figure 3.5). Between them, they capture both the
*persistence over time* and the *repeated content* of sustained harassment.

> **Figure 3.5.** Repetition and intent over a conversation timeline: repeated aggressive messages
> from the same sender to the same target, spread across several days, raise the repetition signal.

### 3.7.2 Intent-to-Harm Scoring

Intent is scored by an `IntentScorer` that reads **severity-weighted cues** in English and Roman
Urdu. Keywords are grouped by severity — death threats (1.0), physical-harm threats (0.8),
intimidation (0.6), degradation (0.5) and social isolation (0.4) — and each group carries both
English and Roman Urdu expressions (for example *"jaan se maar"*, *"tujhe dekh lunga"*,
*"khabardar"*, *"nikamma"*). On top of the keywords, a set of regular-expression **patterns**
catches deliberate constructions such as *"I will hurt you"*, *"you will pay"* and *"just wait"*.
The final score combines the strongest severity matched, a bonus for how many distinct cues appear,
and a pattern bonus, capped at 1.0. The point of the design is that intent shows up not in the
presence of a rude word, but in a threatening, degrading or intimidating *purpose*.

## 3.8 Stage 2 — User-Pair Cyberbullying Classifier

Stage 2 (`train_stage2.py`) makes the final decision for each relationship. The feature vector from
Table 3.3 is assembled, the two users' genders one-hot encoded, and everything standardised using
statistics from the **training split only**, to avoid leakage. A small **multi-layer perceptron** —
fully-connected layers of 32 and 16 units, ReLU and dropout, ending in one logit — maps the vector
to a cyberbullying probability (Figure 3.6).

Because only about 10.4% of pairs are positive, the loss carries a **positive class weight** equal
to the negative-to-positive ratio, so the rare bullying cases are not ignored, and the split is
**stratified** to keep that ratio in both training and test. This stage is intentionally light:
once the features exist, it trains in seconds on a CPU, and the trained model (normalisation
statistics, feature names, weights) is saved to `cb_classifier.pth` for the full pipeline.

> **Figure 3.6.** Stage 2 architecture: a dense network mapping the standardised pair-level vector
> to a cyberbullying probability, trained with a positive-class weight to counter the 10.4%
> positive rate.

## 3.9 Evaluation Metrics

Since both stages work on **imbalanced** data, accuracy on its own is misleading — a model that
always predicts the majority class can post a high accuracy while catching nothing (Section 4.6).
So the system is judged with the full set of metrics in **Table 3.4**, computed from true positives
(TP), false positives (FP), true negatives (TN) and false negatives (FN). **Recall** gets extra
weight, because in a safety setting *missing* a real case (a false negative) usually costs more than
a false alarm.

**Table 3.4.** Evaluation metrics and their definitions.

| Metric | Definition | Interpretation |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness; unreliable under imbalance |
| Precision | TP / (TP + FP) | Of those flagged, how many are truly bullying |
| Recall (Sensitivity) | TP / (TP + FN) | Of all true bullying cases, how many are caught |
| F1-Score | 2 · (Precision · Recall) / (Precision + Recall) | Harmonic mean; balanced single-number summary |

**Confusion matrices** are also reported for both stages to show the full per-class breakdown.

## 3.10 Experimental Setup and Implementation Details

The system is written in **Python** with **PyTorch** and Hugging Face **Transformers**, plus
**scikit-learn** for splitting and metrics and **pandas** for data prep. A single config switch
chooses between `bert-base-multilingual-cased` and `google/muril-base-cased`. The main
hyperparameters are in **Table 3.5**. Stage 1 fine-tuning ran on GPU (Kaggle T4), given the size of
the model and corpus, while Stage 2 trains in seconds on CPU. Data is split into train, validation
and test with a fixed seed for reproducibility, and the validation split drives early stopping in
Stage 1.

**Table 3.5.** Training configuration and hyperparameters.

| Component | Setting |
|---|---|
| Stage 1 text models | m-BERT (`bert-base-multilingual-cased`), MuRIL (`google/muril-base-cased`) |
| Max sequence length | 128 tokens |
| Stage 1 optimiser / learning rate | AdamW / 2 × 10⁻⁵ |
| Stage 1 loss | Binary cross-entropy with class weighting |
| Stage 1 batch size / epochs | 16 / 3 (with early stopping) |
| Stage 2 model | MLP (32 → 16 → 1), ReLU, dropout 0.3 |
| Stage 2 loss | BCE with positive-class weight (neg/pos) |
| Stage 2 optimiser / learning rate | Adam / 1 × 10⁻³ |
| Data split | Stratified train / validation / test, fixed seed |

## 3.11 Web Demonstration Application

To make the system usable and easy to show to a non-technical audience, the trained pipeline is
wrapped in an interactive **web app** (`demo.py`). A user types a message (English, Roman Urdu or
mixed); Stage 1 returns the aggression probability, the intent scorer highlights any harmful cues,
and — given a short conversation — the Stage 2 model reports the relationship-level decision. It
runs in real time, works across languages, and makes the model's reasoning visible instead of
hiding it behind a command line.

## 3.12 Stage 3 — Multimodal Image-and-Text Component

Cyberbullying is not only text. A lot of it travels as **memes**, where an image and a short caption
together carry the hostile message. To reach that modality — and to cover the image side of the
original proposal — a third, **multimodal** component was built, taking an image together with its
overlaid text. It is a separate module, evaluated apart from the conversational system; its results
appear in Section 4.7.

A first version paired a fully fine-tuned **ResNet50** image encoder with **m-BERT** for the text
and fused them with an attention layer (Figure 3.7). On a few thousand memes it overfitted badly and
tended to collapse toward the majority class. That lesson led to a stronger, more stable design
based on **CLIP** (Contrastive Language–Image Pre-training) [38], which learns images and text in a
shared space. In this version the CLIP backbone is **frozen** and only a small head is trained on
top of its image and text embeddings — the standard recipe for limited data, and the one that avoids
the earlier overfitting. Three variants are trained — text-only, image-only and fusion (the
concatenated image and text embeddings) — so the value of combining the two can be measured.

The component is trained on the **Hateful Memes** benchmark [39], which is built so that neither the
image nor the text alone is enough. Since this data shares no users with the conversational set, it
is a **standalone experiment** that complements, but does not replace, the two-stage text system at
the core of this thesis.

> **Figure 3.7.** Initial Stage 3 architecture: ResNet50 (image) and m-BERT / MuRIL (text) features
> combined by an attention layer. This version was later replaced by the frozen-CLIP design of
> Section 4.7.
