# CHAPTER 3

# MATERIALS AND METHODS

## 3.1 Overview of the Proposed Methodology

The central design principle of this work is that **cyberbullying is a property of a
relationship over time, not of a single message**. A model that classifies one message in
isolation can, at best, recognise *aggression*; it cannot recognise the *repetition* and
*intent* that distinguish genuine bullying from an isolated offensive remark
(Sections 1.4 and 2.2). The proposed system therefore separates the problem into **two
stages**, illustrated in **Figure 3.1**:

- **Stage 1 — Message-level aggression.** A multilingual transformer (m-BERT or MuRIL) is
  fine-tuned to decide whether a *single* message — in English or Roman Urdu — is
  aggressive. This is the linguistic building block of the pipeline.
- **Stage 2 — Relationship-level cyberbullying.** For each pair of users, the
  aggressive-message signal is aggregated over their entire conversation history and
  combined with quantitative measures of **repetition**, **intent to harm**, **peerness**
  and **user context** (age, grade, gender). A compact dense neural network then makes the
  final cyberbullying decision for that relationship.

This two-stage structure mirrors the behavioural definition adopted in Chapter 2: Stage 1
supplies the *aggression* pillar at the message level, while Stage 2 reconstructs the
*repetition*, *intent* and *peerness* pillars at the relationship level. The complete
methodology consists of five steps — **data collection and preprocessing, data annotation,
feature extraction, model development (Stages 1 and 2), and evaluation** — each described
in the sections that follow.

> **Figure 3.1.** Overall architecture of the proposed two-stage cyberbullying detection
> system. Stage 1 (m-BERT / MuRIL) classifies individual messages as aggressive; Stage 2
> aggregates these over each user-pair's history and combines them with repetition, intent,
> peerness and context features to output the final cyberbullying label. *(Figure to be
> inserted.)*

## 3.2 Dataset

### 3.2.1 The Comprehensive Cyberbullying Dataset

The primary data source is the comprehensive cyberbullying dataset of **Ejaz, Razi and
Choudhury (2024)** [3], which is distinguished from most public resources by labelling not
only aggression but also the **repetition, peerness and intent to harm** that define
cyberbullying. The dataset is organised as a set of related tables describing a community of
users and the messages exchanged between them:

- **Users** — demographic records for each user, including age, gender, school and grade.
- **Peerness values** — a numeric measure of the social relationship between each pair of
  users, derived from their relative age and grade.
- **Communication data** — the message log between users, with each message carrying an
  aggression label and a timestamp (date and time).
- **CB labels** — the ground-truth cyberbullying label for each user-pair relationship,
  together with aggregate counts (total messages, aggressive count, intent to harm,
  peerness).

Because every message carries a timestamp and is tied to a sender–receiver pair, the
dataset supports exactly the kind of **temporal, relationship-level** analysis that the
definition of cyberbullying requires.

### 3.2.2 Roman Urdu Aggression Data

To make the Stage 1 aggression classifier genuinely **multilingual**, the English
communication data is augmented with a **Roman Urdu** corpus of annotated aggressive and
non-aggressive comments. Each Roman Urdu record contributes its text and a binary
aggression label, allowing the transformer to learn aggression cues in both languages
within a single shared model. The reliability of the Roman Urdu annotations is assessed
separately using Fleiss' Kappa (Section 3.4.2).

### 3.2.3 Message-Level and Pair-Level Views

From these raw tables, an automated preparation pipeline (`prepare_dataset.py`) constructs
the two training tables used by the system, as shown in **Figure 3.2**:

- **`messages.csv` (Stage 1 view).** One row per message, with columns *message*, *label*
  (aggressive / non-aggressive) and *language*. The English communication log and the Roman
  Urdu corpus are concatenated, empty messages are dropped, and exact duplicates are
  removed. The resulting table contains **92,308 messages** — **90,356 English** and
  **1,952 Roman Urdu** — of which **31,389 are labelled aggressive** and **60,919
  non-aggressive** (**Table 3.1**).
- **`pairs.csv` (Stage 2 view).** One row per user-pair, with eighteen columns capturing the
  aggregated behavioural and contextual features of that relationship together with the
  final cyberbullying label. The table contains **9,511 user-pairs**, of which **992 (about
  10.4%) are labelled as cyberbullying** and **8,519 are not** — a substantial class
  imbalance that strongly influences the design and evaluation of Stage 2 (Sections 3.8 and
  3.9).

> **Figure 3.2.** Construction of the message-level and pair-level data views from the raw
> dataset tables. *(Figure to be inserted.)*

**Table 3.1.** Composition of the message-level dataset by language.

| Language | Messages | Share |
|---|---:|---:|
| English | 90,356 | 97.9% |
| Roman Urdu | 1,952 | 2.1% |
| **Total** | **92,308** | **100%** |
| *of which aggressive* | *31,389* | *34.0%* |
| *of which non-aggressive* | *60,919* | *66.0%* |

To make the nature of the data concrete, **Table 3.2** presents representative
annotated messages from both languages. The Roman Urdu examples illustrate the
code-switched, transliterated style discussed earlier; English glosses are provided
in parentheses. (Strong profanity has been masked for presentation.)

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

All message text is passed through a cleaning stage (`TextPreprocessor`) before tokenisation.
The cleaning step removes elements that carry no aggression signal and would otherwise add
noise:

- **URLs** (`http(s)://…`, `www.…`) are stripped.
- **HTML tags** are removed.
- **User mentions** (`@username`) are removed, since the identity of the mentioned account
  is irrelevant to whether the message is aggressive.
- **Emojis and stray special characters** are replaced with spaces.
- **Whitespace** is normalised so that runs of spaces collapse to a single space.

#### 3.3.2 Roman Urdu Handling and Tokenisation

Roman Urdu poses a particular challenge because it has **no standardised spelling**: the
same word may be written in many different ways. A light normalisation step lower-cases the
text and reduces some common repeated-vowel spelling variations (for example collapsing
doubled vowels) so that superficially different spellings of the same word are brought
closer together. This is deliberately conservative; the heavy lifting of cross-language and
cross-spelling generalisation is left to the multilingual transformer, whose sub-word
tokeniser and shared representation space are far better suited to spelling variation than a
fixed rule set or lexicon would be. Each cleaned message is then tokenised with the model's
own pretrained tokeniser (WordPiece for m-BERT, the MuRIL tokeniser for MuRIL) and truncated
or padded to a fixed maximum length.

## 3.4 Data Annotation and Inter-Annotator Agreement

### 3.4.1 Annotation Scheme

The dataset is annotated against the four behavioural pillars introduced in Chapter 1:

- **Aggression** — a binary label on each message indicating whether it is hostile, abusive
  or offensive.
- **Repetition** — derived from the conversation history of each user-pair, capturing how
  many aggressive acts occur and how persistently they recur over time.
- **Intent to harm** — a measure of the deliberate purpose to threaten, degrade, intimidate
  or isolate the victim.
- **Peerness** — the social relationship and power balance between the two users, derived
  from their relative age and grade.

The final per-relationship **cyberbullying label (CB_Label)** is the ground truth that
Stage 2 is trained to predict.

The Roman Urdu portion of the data was supplied with a **three-level (0/1/2)** annotation
scale — corresponding to neutral, positive and hostile/negative tone — assigned
independently by three annotators. For the aggression task, only the **hostile (level 2)**
class is treated as aggressive; each annotator's rating is mapped to a binary
"aggressive" vote (`rating == 2`) and a **majority vote** (at least two of three) of these
determines the final aggression label. This consensus mapping yields 612 aggressive Roman
Urdu messages and is the basis of the message-level labels used for Stage 1.

### 3.4.2 Fleiss' Kappa Agreement

Because labelling abusive language involves subjective judgement, the **reliability** of the
annotations must be demonstrated rather than assumed. For the Roman Urdu data, each item was
labelled by **three annotators**, and agreement among them is quantified using **Fleiss'
Kappa** (`fleiss_kappa.py`), the standard chance-corrected measure of agreement among
multiple raters on categorical labels. For two categories (bullying / non-bullying) and
three raters, a ratings matrix is built in which each row records how many of the three
annotators assigned each category to that item, and Kappa is computed as

  κ = (P̄ − Pₑ) / (1 − Pₑ),

where P̄ is the mean observed agreement across items and Pₑ is the agreement expected by
chance. The resulting value is interpreted on the conventional scale (slight, fair,
moderate, substantial, almost-perfect agreement), and the proportion of items with unanimous
versus majority agreement is reported alongside it (Section 4.3). Reporting Kappa
establishes that the labels driving the model are consistent and trustworthy.

## 3.5 Feature Extraction

### 3.5.1 Textual Features (m-BERT / MuRIL Embeddings)

Textual meaning is captured by a **multilingual transformer**. Two models are evaluated:

- **m-BERT (`bert-base-multilingual-cased`)**, pretrained on the Wikipedia text of more than
  one hundred languages, providing a shared multilingual representation suited to
  code-switched English/Roman Urdu text.
- **MuRIL (`google/muril-base-cased`)**, pretrained specifically on South Asian languages and
  on transliterated text, making it directly relevant to Roman Urdu.

Both produce a 768-dimensional contextual embedding for each message. Unlike Bag-of-Words or
lexicon features, these embeddings encode word order, context and cross-lingual relationships,
allowing the model to interpret meaning that depends on how words are used rather than merely
which words are present.

### 3.5.2 Behavioural and Contextual Features

At the relationship level, each user-pair is summarised by a vector of **behavioural and
contextual features** computed from its conversation history (**Table 3.3**). These include
the volume of communication, the amount and proportion of aggression, temporal repetition
features (how many distinct days aggression occurred on, and over how long a span), the
intent-to-harm score, the peerness score, and the demographic context of both participants.
Together these convert a relationship's entire history into a single fixed-length feature
vector that Stage 2 can classify.

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
classification, as shown in **Figure 3.4**. A message is tokenised and passed through the
m-BERT/MuRIL encoder; the pooled `[CLS]` representation (768 dimensions) is taken as the
message embedding, passed through a dropout layer for regularisation, and projected by a
single linear layer to one logit. A sigmoid converts this logit to the probability that the
message is aggressive. The entire network — encoder plus classification head — is fine-tuned
end-to-end using the binary cross-entropy objective.

> **Figure 3.4.** Stage 1 architecture: a multilingual transformer encoder (m-BERT / MuRIL)
> followed by dropout and a single linear classification head producing an aggression
> probability. *(Figure to be inserted.)*

### 3.6.2 Class Imbalance Handling

Aggressive messages are a minority of the corpus (about 35%; Table 3.1). Left uncorrected,
this imbalance encourages a model to favour the majority (non-aggressive) class. To counter
this, **class weighting** is applied in the loss function so that errors on the minority
(aggressive) class are penalised more heavily, and **precision, recall and F1-score** —
rather than accuracy alone — are used to monitor training. This design decision is a direct
response to the majority-class-collapse failure observed in an earlier version of the system
(Sections 1.6 and 4.6).

## 3.7 Quantifying Repetition and Intent

A defining contribution of this work is that repetition and intent are **measured
quantitatively**, not assumed. Two scoring components (`scoring.py`) operationalise these
pillars; the temporal repetition features used by Stage 2 are additionally derived directly
from the timestamped conversation log.

### 3.7.1 Repetition Scoring

Repetition is treated as the **recurrence of aggression between the same pair of users over
time**. Two complementary signals capture it. First, from the conversation log, the pipeline
computes how many **distinct days** aggression occurred on (`aggression_active_days`) and the
**span of days** between the first and last aggressive message (`aggression_span_days`); a
`repetition_flag` marks relationships with two or more aggressive messages. Of the 9,511
user-pairs, **5,568 exhibit repeated aggression** under this flag. Second, a `RepetitionScorer`
provides a content-based measure: it compares a user's messages using **Jaccard similarity**
over their word sets and counts near-duplicate pairs above a similarity threshold, weighting
the result by message frequency so that frequent, repeated attacks score highly while a
single message scores zero (**Figure 3.5**). Together these capture both the *temporal
persistence* and the *content repetition* that characterise sustained harassment.

> **Figure 3.5.** Repetition and intent scoring over a conversation timeline: repeated
> aggressive messages from the same sender to the same target, persisting across multiple
> days, raise the repetition signal. *(Figure to be inserted.)*

### 3.7.2 Intent-to-Harm Scoring

Intent is estimated by an `IntentScorer` that recognises **severity-weighted linguistic
cues** in both English and Roman Urdu. Keywords are grouped into categories ordered by
severity — death threats (weight 1.0), threats of physical harm (0.8), intimidation (0.6),
degradation (0.5) and social isolation (0.4) — and each category includes both English and
Roman Urdu expressions (for example *"jaan se maar"*, *"tujhe dekh lunga"*, *"khabardar"*,
*"nikamma"*). In addition, a set of regular-expression **patterns** detects deliberate
constructions such as *"I will hurt you"*, *"you will pay"* and *"just wait"*. The final
intent score combines the maximum severity matched, a bonus for the number of distinct cues,
and a pattern-match bonus, capped at 1.0. This bilingual, severity-aware design recognises
that intent is conveyed not by the mere presence of a rude word but by threatening,
degrading or intimidating *purpose*.

## 3.8 Stage 2 — User-Pair Cyberbullying Classifier

Stage 2 (`train_stage2.py`) makes the final cyberbullying decision for each relationship. The
pair-level feature vector of Table 3.3 is assembled, with the two users' genders one-hot
encoded, and standardised using statistics computed on the **training split only** (to avoid
information leakage). A compact **multi-layer perceptron (MLP)** — fully-connected layers of
32 and 16 units with ReLU activations and dropout, ending in a single logit — maps this
vector to the probability of cyberbullying (**Figure 3.6**).

Because only about 10.4% of pairs are positive, the loss uses a **positive class weight**
equal to the ratio of negative to positive examples, so that the rare cyberbullying cases are
not ignored. The data is split with **stratification** to preserve this ratio across the
training and test sets. This stage is deliberately lightweight: once the pair features are
available, it trains in seconds on a CPU, and the trained classifier (mean/standard-deviation
normalisation statistics, feature names and weights) is saved to `cb_classifier.pth` for use
in the end-to-end pipeline.

> **Figure 3.6.** Stage 2 architecture: a dense neural network mapping the standardised
> pair-level feature vector to a cyberbullying probability, trained with a positive-class
> weight to counter the 10.4% positive rate. *(Figure to be inserted.)*

## 3.9 Evaluation Metrics

Because both stages operate on **imbalanced** data, accuracy alone is an inadequate and
potentially misleading measure: a model that predicts the majority class for every input can
report a high accuracy while detecting nothing of interest (Section 4.6). The system is
therefore evaluated using the full set of metrics defined in **Table 3.4**, computed from the
counts of true positives (TP), false positives (FP), true negatives (TN) and false negatives
(FN). **Recall** is given particular weight, since in a safety-critical setting the cost of
*missing* a genuine case of cyberbullying (a false negative) is generally higher than that of
a false alarm.

**Table 3.4.** Evaluation metrics and their definitions.

| Metric | Definition | Interpretation |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Overall correctness; unreliable under imbalance |
| Precision | TP / (TP + FP) | Of those flagged, how many are truly bullying |
| Recall (Sensitivity) | TP / (TP + FN) | Of all true bullying cases, how many are caught |
| F1-Score | 2 · (Precision · Recall) / (Precision + Recall) | Harmonic mean; balanced single-number summary |

In addition, **confusion matrices** are reported for both stages to expose the full breakdown
of correct and incorrect predictions per class.

## 3.10 Experimental Setup and Implementation Details

The system is implemented in **Python** using **PyTorch** and the Hugging Face
**Transformers** library, with **scikit-learn** for data splitting and metrics and
**pandas** for data preparation. The two transformer variants — `bert-base-multilingual-cased`
and `google/muril-base-cased` — are selected through a single configuration switch. The main
training hyperparameters are summarised in **Table 3.5**. Stage 1 fine-tuning was performed on
GPU (Kaggle T4) owing to the size of the transformer and the corpus, while Stage 2 trains in
seconds on CPU. Data is partitioned into training, validation and test splits with a fixed
random seed for reproducibility, and the validation split is used for early stopping during
Stage 1 fine-tuning.

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

To make the system usable and to demonstrate it to a non-technical audience, the trained
pipeline is wrapped in an interactive **web demonstration** (`demo.py`). A user enters a
message (in English, Roman Urdu or code-switched text); Stage 1 returns the aggression
probability, the intent scorer highlights any harmful cues, and — given a conversation
history — the Stage 2 model reports the relationship-level cyberbullying decision. This
demonstrates real-time, multilingual operation and makes the model's reasoning visible rather
than hidden behind a command line.

## 3.12 Stage 3 — Multimodal Image-and-Text Component

Cyberbullying online is not confined to plain text: a large share of abusive content
takes the form of **memes**, in which an image and a short caption together convey a
hostile or humiliating message. To extend the framework toward this modality — and to
address the image-based dimension of the original proposal — a third, **multimodal**
classifier was implemented in which each sample consists of an image together with its
overlaid text. This component is exploratory and is evaluated separately from the
conversational text system; its results and limitations are reported in Section 4.7.

The architecture (`fusion_model.py`) encodes the two modalities with complementary
backbones and combines them through attention (**Figure 3.7**):

- **Image branch.** A **ResNet50** convolutional network, pretrained on ImageNet,
  encodes the meme image into a 2048-dimensional visual feature vector.
- **Text branch.** The same multilingual transformer used in Stage 1
  (**m-BERT / MuRIL**) encodes the meme's text into a 768-dimensional embedding.
- **Attention fusion.** Each modality is projected to a common 512-dimensional space,
  and a learned **attention layer** computes a weighted combination of the two, so the
  model can rely on whichever modality is more informative for a given sample. A dense
  classification head then predicts **bullying / not bullying**.

The component was trained on the publicly available **Memotion** meme dataset (6,992
memes, of which 4,279 carry an offensive label that is treated here as the positive
class). Because the meme dataset shares no users with the conversational data, it is a
**standalone experiment** rather than a fusion over the same samples; it complements,
but does not replace, the two-stage text system that forms the core of this thesis.

> **Figure 3.7.** Stage 3 multimodal architecture: ResNet50 (image) and m-BERT / MuRIL
> (text) features combined by an attention layer for a binary bullying decision.
> *(Figure to be inserted.)*
