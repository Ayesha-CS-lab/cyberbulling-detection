# CHAPTER 1

# INTRODUCTION

## 1.1 Background

Over the past two decades social media has transformed the way people communicate,
share opinions and maintain relationships across the world. Platforms such as Facebook,
Instagram, Twitter (X), WhatsApp and YouTube allow billions of users to interact
instantly, regardless of distance. This connectivity has brought enormous social and
economic benefits, but it has also created new avenues for harmful behaviour. Among the
most serious of these is **cyberbullying** — the use of electronic communication to
repeatedly and deliberately harm, threaten, intimidate or humiliate another person.

Unlike traditional, face-to-face bullying, cyberbullying is not bounded by physical
location or time. It can follow a victim from the schoolyard into their home, continue
around the clock, reach a wide audience within seconds, and — because of the relative
anonymity of online platforms — be carried out by aggressors who never have to confront
their victims directly. Offensive messages, rumours, threats and humiliating content,
once posted, are difficult to remove and may resurface repeatedly. As a result,
cyberbullying frequently has a more persistent and far-reaching psychological impact
than its offline counterpart.

The scale of the problem has grown in step with the scale of social media itself. Large
numbers of adolescents and young adults report having either experienced or witnessed
cyberbullying, and the proportion of online interactions that involve some form of
abusive or harassing content remains substantial. Because the volume of messages
generated on modern platforms is far too large for human moderators to review manually,
there is a clear and pressing need for **automatic detection systems** that can identify
cyberbullying reliably and at scale.

The work presented in this thesis addresses this need with a specific focus that has
received comparatively little attention: the detection of cyberbullying in the
**multilingual, code-switched text** typical of South Asian — and particularly
Pakistani — social media, where users freely mix Urdu, Roman Urdu and English within a
single conversation.

## 1.2 Cyberbullying and Its Effects

Cyberbullying is defined in the research literature not as a single offensive act but as
**an aggressive, intentional act carried out repeatedly and over time, by a group or an
individual, using electronic forms of contact, against a victim who cannot easily defend
themselves** [3]. Three properties recur in almost every accepted definition: the
behaviour is *aggressive or harmful*, it is *repeated* rather than isolated, and it is
*intentional*. A single rude comment made in the heat of an argument is unpleasant, but
it is not, on its own, cyberbullying; the same comment directed at the same victim
repeatedly and with evident intent to cause harm is.

The consequences of cyberbullying for its victims are severe and well documented. Victims
commonly experience depression, anxiety, fear, lowered self-esteem and feelings of
isolation. These psychological effects are frequently accompanied by physical symptoms,
disturbed sleeping and eating patterns, declining academic performance, withdrawal from
school and loss of interest in everyday activities. In the most tragic cases, sustained
cyberbullying has been associated with self-harm and suicidal ideation. Because the harm
is cumulative — building over repeated incidents rather than resulting from any single
message — early and accurate detection of *ongoing* bullying relationships is of
particular importance.

Social media providers have introduced a range of countermeasures, including reporting
tools, blocking and muting features, and dedicated safety resources. However, the
majority of these mechanisms are **reactive**: they operate only after abusive content
has already been posted and, in most cases, only after a human user has flagged it.
There remains a strong case for **proactive, automated detection** that can identify
harmful behaviour as it unfolds, rather than after the damage has been done.

## 1.3 The Multilingual Challenge: Urdu, Roman Urdu and Code-Switching

The overwhelming majority of research on automatic cyberbullying detection has focused on
the **English language**. This leaves a substantial gap for the many millions of users
who communicate online in other languages. The gap is especially acute in South Asia,
where the volume of social media activity is enormous and where online communication has
distinctive linguistic characteristics that English-centric models are not equipped to
handle.

In Pakistan and neighbouring regions, online conversations are rarely conducted in a
single, clean language. Instead they exhibit two intertwined phenomena:

- **Roman Urdu** — Urdu written using the Latin (English) alphabet rather than the
  Nastaʿlīq script (for example, *"tum bohat buray ho"* instead of "تم بہت برے ہو").
  Roman Urdu has no standardised spelling, so the same word may be written in many
  different ways, which makes lexicon- and dictionary-based methods particularly
  unreliable.
- **Code-switching** — the mixing of two or more languages within a single sentence or
  conversation (for example, *"yaar tum to total loser ho, kabhi sudhroge nahi"*).
  Real users move fluidly between Urdu, Roman Urdu and English, often within the same
  message.

These properties pose serious problems for conventional approaches. A model trained only
on English text cannot interpret Roman Urdu; a model trained on a single language cannot
follow a conversation that switches between languages; and a lexicon of "bad words"
cannot keep pace with the spelling variation and slang of an unstandardised, rapidly
evolving online dialect. Addressing cyberbullying for Urdu-speaking communities therefore
requires models that are **multilingual by design**. Recent **multilingual transformer
models** — in particular **Multilingual BERT (m-BERT)** and **MuRIL (Multilingual
Representations for Indian Languages)** — are well suited to this task, because they
learn shared representations across many languages and can relate words and meaning
across language boundaries.

## 1.4 Beyond Aggression: Repetition, Intent and Peerness

A second, equally important gap concerns *what* existing systems actually detect. The
great majority of cyberbullying and hate-speech detection systems — including most of the
work that has been done on Roman Urdu — treat the problem as **aggression detection**:
given a single message, decide whether it is aggressive or abusive. This is a useful
building block, but it is not the same as detecting cyberbullying. As Section 1.2
established, cyberbullying is defined by **repetition** and **intent** as well as
aggression. A system that flags every aggressive message as "cyberbullying" will
inevitably misclassify one-off insults, jokes between friends and momentary outbursts as
bullying, while having no way to recognise a sustained campaign of harassment as a single,
escalating pattern.

This thesis adopts the more complete, behavioural view of cyberbullying captured by
recent work that incorporates **aggression, repetition, peerness and intent to harm**
[3]. The four concepts are defined as follows:

- **Aggression** — whether an individual message is hostile, abusive or offensive.
- **Repetition** — whether aggressive messages from the same sender to the same target
  recur and persist over time, rather than occurring once.
- **Intent to harm** — whether the language conveys a deliberate purpose to threaten,
  degrade, intimidate or isolate the victim.
- **Peerness** — the social relationship and power balance between the two users (for
  example their relative age or grade), which helps distinguish bullying from banter
  between equals.

**Figure 1.1** illustrates the relationship between these elements: aggression is the raw
signal at the level of a single message, while repetition and intent — interpreted in the
light of peerness — elevate a series of aggressive messages into genuine cyberbullying.
The framework developed in this thesis is built directly around this distinction.

> **Figure 1.1.** The behavioural pillars of cyberbullying. A single aggressive message
> (Stage 1) becomes cyberbullying only when it is *repeated over time*, carries *intent to
> harm*, and is interpreted in the context of the *peer relationship* between the users
> (Stage 2). *(Figure to be inserted.)*

## 1.5 Problem Statement

Although deep-learning methods for detecting online abuse have advanced rapidly, several
limitations continue to restrict their usefulness for real, multilingual social media:

1. **English-centric design.** Most existing detection systems are built and evaluated on
   English text and do not transfer to Urdu or Roman Urdu, leaving a large population of
   users unprotected.

2. **No support for code-switching.** There is a shortage of large, labelled datasets
   that combine Urdu, Roman Urdu and English, and most models cannot process the mixed,
   code-switched text that dominates South Asian social media.

3. **Aggression treated as cyberbullying.** The majority of systems classify a single
   message as aggressive or not, and equate aggression with cyberbullying. They therefore
   misclassify isolated aggressive remarks as bullying and cannot recognise *repeated*,
   *intentional*, *targeted* harassment as the connected behaviour that it is.

4. **Behaviour assessed without context.** Existing approaches rarely combine the textual
   signal with behavioural and relational context — how often the aggression recurs, how
   harmful the intent is, and the social relationship between the users — all of which are
   essential to a correct cyberbullying decision.

This thesis addresses these problems through an integrated, **multilingual, context-aware,
two-stage framework** that detects message-level aggression across Urdu, Roman Urdu and
English, and then combines it with measures of repetition, intent and peerness to make a
final, relationship-level cyberbullying decision.

## 1.6 Motivation

The motivation for this research is both social and technical. **Socially**, the people
most affected by the English-language bias of current systems are precisely those in
multilingual, lower-resource communities — including Urdu-speaking users — who receive the
least protection despite being heavily active online. Building a detector that understands
Roman Urdu and code-switched text is therefore a step toward a more equitable online
environment.

**Technically**, the project is motivated by the observation that high reported accuracy
can be deeply misleading when a dataset is imbalanced. In a realistic setting, bullying
messages are far rarer than ordinary ones; a naïve model can therefore achieve a high
"accuracy" simply by predicting that nothing is ever bullying — while in fact detecting
none of it. An early version of the system developed in this thesis exhibited exactly this
**majority-class collapse**, reporting an apparently respectable accuracy while failing to
identify a single genuine case (Section 4.6). Diagnosing and correcting that failure
motivated the central design decisions of this work: the use of class weighting,
appropriate evaluation metrics (precision, recall and F1 rather than accuracy alone), and
a two-stage architecture that models the behaviour of cyberbullying rather than the
surface form of a single message.

## 1.7 Aim and Objectives

The **aim** of this project is to design, implement and evaluate a **context-aware,
multilingual deep-learning framework that accurately detects cyberbullying in
code-switched Urdu, Roman Urdu and English text** by modelling aggression, repetition and
intent to harm within the social context of a user relationship.

To achieve this aim, the following **objectives** are pursued:

1. To assemble and preprocess a multilingual dataset of social-media messages containing
   Urdu, Roman Urdu and English, and to organise it into a **message-level** view (for
   aggression) and a **user-pair view** (for cyberbullying).
2. To establish the reliability of the dataset's labels for aggression, repetition, intent
   and peerness, and to assess inter-annotator agreement using **Fleiss' Kappa**.
3. To fine-tune multilingual transformer models — **m-BERT** and **MuRIL** — for
   message-level aggression detection, applying class weighting to counter class
   imbalance.
4. To design **quantitative scoring methods** for repetition (frequency and persistence
   of aggression over time) and intent to harm (severity-weighted linguistic cues in both
   English and Roman Urdu).
5. To develop a **second-stage classifier** that integrates the aggression signal with
   repetition, intent, peerness and user-context features to produce a final,
   relationship-level cyberbullying decision.
6. To evaluate the framework using accuracy, precision, recall and F1-score, to compare
   m-BERT and MuRIL against conventional baselines (SVM and LSTM), and to document and
   correct the majority-class-collapse failure mode.
7. To deploy the trained system as an **interactive web application** that demonstrates
   real-time, multilingual cyberbullying detection.

## 1.8 Research Questions

The investigation is guided by the following research questions:

- **RQ1.** Can a multilingual transformer (m-BERT or MuRIL) reliably detect message-level
  aggression across English and Roman Urdu, including code-switched text, and which model
  performs better?
- **RQ2.** Does a two-stage design — separating message-level aggression from
  relationship-level cyberbullying — recover the repetition and intent dimensions of
  cyberbullying that single-stage aggression classifiers fail to capture?
- **RQ3.** Can repetition and intent to harm be quantified from conversation history in a
  way that meaningfully improves the final cyberbullying decision?
- **RQ4.** How does the corrected, class-balanced model compare with an earlier model that
  suffered majority-class collapse, and what does this contrast reveal about the correct
  evaluation of imbalanced detection tasks?

## 1.9 Scope of the Study

This study focuses on the **text-based** detection of cyberbullying in three languages —
**English, Roman Urdu and Urdu** — within the conversational, user-to-user setting
captured by the dataset described in Chapter 3. The scope covers the complete pipeline:
data assembly and preprocessing, label-reliability assessment, multilingual aggression
detection, repetition and intent scoring, relationship-level cyberbullying classification,
evaluation against baselines, and deployment as a web demonstration.

The study is deliberately **text-only**. Although the original proposal also envisaged an
image modality, the multilingual, code-switched behaviour that defines this problem — and
the comprehensive labelled data for aggression, repetition, intent and peerness — exist in
the text domain, which is therefore where the contribution of this work is concentrated.
Image-based and fully multimodal extension is identified as future work (Section 5.4).
The study does **not** address sarcasm detection, context or sentiment beyond the features
described, real-time deployment at platform scale, or the legal and privacy frameworks
required for operational use; these are acknowledged as important but lie outside the
boundaries of this academic project.

## 1.10 Significance and Contributions

The principal contributions of this thesis are:

1. **A multilingual, code-switching-aware aggression detector** built on m-BERT / MuRIL
   and trained on a combined English and Roman Urdu corpus, extending cyberbullying
   research beyond its usual English-only setting.
2. **A two-stage, behaviourally complete framework** that separates message-level
   *aggression* from relationship-level *cyberbullying*, thereby modelling the
   **repetition**, **intent** and **peerness** that define genuine bullying rather than
   equating it with a single aggressive message.
3. **Quantitative repetition and intent scoring** methods that operate over the
   conversation history of a user-pair and function in both English and Roman Urdu.
4. **An honest, imbalance-aware evaluation**, including the explicit diagnosis and
   correction of a majority-class-collapse failure mode — an instructive contrast between
   a model that merely *appears* to work and one that genuinely does.
5. **A deployable web application** that demonstrates real-time, multilingual cyberbullying
   detection across Urdu, Roman Urdu and English.

## 1.11 Organisation of the Thesis

The remainder of this thesis is structured as follows:

- **Chapter 2 — Literature Review** surveys the definition of cyberbullying, traditional
  machine-learning and lexicon-based methods, deep-learning and transformer approaches,
  cyberbullying and aggression detection in Roman Urdu, the code-switching problem,
  behavioural and contextual features, and dataset and annotation practices, concluding
  with the research gap that this work addresses.
- **Chapter 3 — Materials and Methods** describes the dataset and its message-level and
  pair-level views, the preprocessing and Roman Urdu normalisation pipeline, the
  annotation scheme and Fleiss' Kappa assessment, feature extraction, the Stage 1
  aggression model, the repetition and intent scoring methods, the Stage 2 cyberbullying
  classifier, the evaluation metrics, the experimental setup and the web demonstration.
- **Chapter 4 — Results and Discussion** presents the dataset statistics, the
  inter-annotator agreement results, the Stage 1 and Stage 2 results, the comparison with
  the earlier collapsed model, the end-to-end pipeline behaviour and the web application,
  and discusses the findings in the context of existing literature.
- **Chapter 5 — Conclusion and Future Work** summarises the contributions of the study,
  acknowledges its limitations and proposes directions for future research, including the
  multimodal extension.
