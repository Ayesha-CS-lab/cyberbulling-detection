# CHAPTER 2

# LITERATURE REVIEW

## 2.1 Introduction

Research on the automatic detection of cyberbullying sits at the intersection of natural
language processing, machine learning and online-safety studies. Over the past decade the
field has progressed from simple keyword filtering and classical machine-learning
classifiers through to deep neural networks and, most recently, large multilingual
transformer models. This chapter reviews the work most relevant to the present study. It
begins by examining how cyberbullying is defined in the literature and why that definition
matters for system design. It then surveys traditional machine-learning and lexicon-based
detection methods, the shift toward deep learning and transformer models, and the smaller
but growing body of work on Urdu and Roman Urdu. It considers the specific difficulty of
code-switched text, the role of behavioural and contextual features, and the practical
questions of dataset creation and annotation reliability. The chapter concludes by
identifying the research gap that motivates this thesis.

## 2.2 Defining Cyberbullying: A Behavioural View

Before a system can detect cyberbullying, it must be clear about what cyberbullying *is*.
The most widely cited definitions describe it as **"an aggressive, intentional act carried
out by a group or an individual, using electronic forms of contact, repeatedly and over
time, against a victim who cannot easily defend themselves"** [3]. Stewart and Petermann
(2018) similarly emphasise that cyberbullying is the *deliberate, repetitive and harmful*
use of information and communication technology (ICT), highlighting the psychological
motive and the frequency of abuse as defining characteristics rather than incidental ones.

Two implications follow from these definitions. First, **a single offensive message is not,
by itself, cyberbullying** — repetition and intent are constitutive, not optional. Second,
cyberbullying is fundamentally a property of a *relationship over time* between an
aggressor and a victim, not of an isolated piece of text. Despite this, much of the
detection literature reduces the problem to the binary classification of single messages
as aggressive or not. This thesis follows the more complete operationalisation advanced by
Ejaz, Razi and Choudhury (2024) [3], who construct a dataset and framework that explicitly
incorporate **aggressive texts, repetition, peerness and intent to harm** — the four
elements adopted in the present work.

## 2.3 Traditional and Machine-Learning Approaches

The earliest computational approaches to cyberbullying and abusive-language detection
relied on **classical supervised machine learning**. The common pipeline consisted of
gathering and labelling a dataset, preprocessing the text, converting it into features
(typically Bag-of-Words or *n*-gram term-frequency vectors, sometimes weighted by TF–IDF),
and training a classifier such as Support Vector Machine (SVM), Naïve Bayes (NB), Logistic
Regression (LR) or Stochastic Gradient Descent (SGD).

A number of representative studies illustrate this paradigm and its multilingual reach.
Working on Arabic tweets, Haidar et al. applied SVM and Naïve Bayes and reported F1-scores
of around 0.92 and 0.90 respectively. Özel et al. classified Turkish text from Twitter and
Instagram using decision trees, SVM, Multinomial Naïve Bayes and *k*-Nearest Neighbours,
finding that incorporating emoticons as features improved accuracy and that Naïve Bayes
performed best at around 84%. Nurrahmi et al. detected cyberbullying in Indonesian using
SVM and additionally tracked each user's history to estimate credibility, reporting an
F1-score of roughly 0.67. Pawar (2019), in a multilingual system covering English, Hindi
and Marathi, combined Multinomial Naïve Bayes, SGD and Logistic Regression within a
distributed, collaborative architecture and showed that machine-learning models
consistently outperformed lexicon-based ones across all three languages.

These approaches established important baselines, but they share two limitations. First,
they generally treat text as a **bag of independent words**, ignoring grammar, negation
and word order, so meaning that depends on context is lost. Second, almost all of this work
is monolingual and the majority of it is English; performance degrades sharply on the
informal, misspelled and code-switched text typical of real social media.

## 2.4 Lexicon-Based Approaches

A parallel line of work uses **lexicon-based** methods, which rely on a curated dictionary
of abusive, profane or offensive terms and detect bullying by matching message text against
this corpus. Chen et al. proposed the **Lexical Syntactic Feature (LSF)** method for
sentence-level offensive-language detection and reported high precision and recall on
English text. Kontostathis et al. analysed the specific vocabulary used by cyberbullies and
constructed queries from these terms to retrieve bullying content.

Lexicon-based methods are transparent and require no training data, but they are brittle.
Their coverage is limited to the words in the dictionary; they cannot interpret novel
slang, sarcasm or obfuscated spelling; and a single unrecognised or misspelled word can
flip the score of an entire message. These weaknesses are especially severe for **Roman
Urdu**, which has no standardised spelling — a single abusive word may appear in many
written forms, none of which a fixed lexicon can be guaranteed to contain. For this reason
the present work treats lexicon matching only as a supporting signal (within the intent
scorer) rather than as a primary classifier.

## 2.5 Deep Learning and Transformer Models (BERT, m-BERT, MuRIL)

The limitations of hand-engineered features motivated a shift toward **deep learning**,
which learns feature representations directly from data. Convolutional Neural Networks
(CNNs), Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks were
applied to abusive-language detection and improved on classical baselines by capturing
local patterns and sequential dependencies in text. Khan et al. (2022), for example,
applied a CNN–LSTM architecture to English and Roman Urdu sentiment data, demonstrating
that deep models could be trained on Roman Urdu text directly.

The introduction of the **Transformer** architecture and, in particular, **BERT
(Bidirectional Encoder Representations from Transformers)** by Devlin et al. (2018) marked
a turning point. By pre-training a bidirectional language model on very large corpora and
fine-tuning it on a downstream task, BERT captures deep contextual meaning that earlier
models could not. El Koshiry (2024) found that transformer models outperform traditional
deep-learning architectures such as CNNs and RNNs for cyberbullying detection.

For multilingual settings, two transformer variants are particularly relevant and are the
models adopted in this thesis:

- **Multilingual BERT (m-BERT)** is pre-trained on the Wikipedia text of more than one
  hundred languages, learning a shared representation space in which words from different
  languages are related. This makes it a natural fit for code-switched text.
- **MuRIL (Multilingual Representations for Indian Languages)**, introduced by Khanuja et
  al. (2021), is pre-trained specifically on South Asian languages and on **transliterated**
  text, which is directly relevant to Roman Urdu. Das et al. (2022) and related work report
  strong results for MuRIL on low-resource South Asian tasks.

Critically, however, this body of work applies transformers **almost exclusively to
aggression or hate-speech classification of single messages**. The models are rarely, if
ever, used to capture the *repetition* and *intent* that the definition of cyberbullying
requires — a gap between linguistic capability and behavioural understanding that this
thesis sets out to close.

## 2.6 Cyberbullying and Aggression Detection in Roman Urdu

A focused stream of research has begun to address Urdu and Roman Urdu directly. Dewani et
al. (2021, 2023), Anwar and Anwar (2022) and Rasheed et al. (2022) all developed NLP and
deep-learning models for detecting **aggressive or abusive Roman Urdu comments**, achieving
high accuracy for aggression detection. Razi and Ejaz (2024) advanced a multilingual
cyberbullying detection system for Urdu, Roman Urdu and English content, training on a
multilingual dataset and reporting high accuracy.

While these works are valuable and establish that Roman Urdu abuse can be detected, they
share a consistent limitation: they **equate cyberbullying with aggression**. Each treats
the task as the classification of a single comment as aggressive or not, and none
incorporates repetition or intent to harm. They therefore capture only one of the three
behavioural pillars identified in Section 2.2, and cannot distinguish a one-off insult from
a sustained campaign of harassment.

## 2.7 The Code-Switching Problem in South Asian Social Media

Several authors highlight **code-switching** — the mixing of two or more languages within a
single message — as a central, unresolved challenge. Khan and Qureshi (2022) and Akhter et
al. (2020) argue that real social-media posts routinely combine Urdu, Roman Urdu and
English, and that the datasets available for training rarely reflect this reality: most are
restricted to a single language, which limits the generalisation of any model trained on
them. They identify the **creation of genuinely multilingual, code-switched datasets** as a
key prerequisite for progress.

This observation has two consequences for system design. First, monolingual models and
fixed lexicons are structurally unsuited to code-switched input. Second, a model must be
exposed during training to the same kind of mixed text it will encounter at inference time.
The present work responds to both points by training its aggression model on a combined
English and Roman Urdu corpus using multilingual transformers that share a representation
space across languages.

## 2.8 Behavioural and Contextual Features

A smaller body of work recognises that text alone is insufficient and incorporates
**behavioural and contextual** signals. Ting et al. mined social-network features —
keywords, social-network analysis and sentiment — and found sentiment, as a proxy for
intent, to be the single most informative feature. Nurrahmi et al. tracked the ratio of
bullying to non-bullying messages per user in order to estimate the credibility of a
potential aggressor. Silva et al.'s *BullyBlocker* maintained a 60-day record of a user's
activity rather than judging messages in isolation.

These studies point toward the importance of features that this thesis makes central:
**repetition** (how often and how persistently aggression recurs), **intent** (the
severity and purpose of the language) and **peerness** (the social relationship and power
balance between two users). Stewart and Petermann's emphasis on repetition and motive, and
Hasan's (2023) call for *intent detection* in identifying true cyberbullying, reinforce the
need to combine the textual signal with these relational features rather than relying on
text classification alone.

## 2.9 Datasets, Annotation and Inter-Annotator Agreement

Progress in this field is constrained by the availability of high-quality labelled data.
Hasan (2023) identifies the scarcity of datasets for low-resource languages such as Urdu and
Hindi as a fundamental obstacle, noting that much of the existing work remains theoretical
for lack of a suitable corpus. The dataset used in this thesis, derived from the
comprehensive cyberbullying resource of Ejaz, Razi and Choudhury (2024) [3], is notable
precisely because it labels not only aggression but also repetition, peerness and intent at
the level of user relationships.

Where labels are assigned by human annotators, their **reliability** must be established.
The standard instrument for measuring agreement among multiple annotators on categorical
labels is **Fleiss' Kappa**, which corrects observed agreement for the agreement expected by
chance. Reporting inter-annotator agreement is essential for demonstrating that a dataset's
labels are consistent and trustworthy; accordingly, this study assesses agreement using
Fleiss' Kappa (Section 3.4).

## 2.10 Research Gap

The reviewed literature reveals a consistent and compound gap, summarised in **Table 2.1**:

1. **English dominance.** The overwhelming majority of cyberbullying detection research
   targets English, leaving Urdu- and Roman-Urdu-speaking users underserved.
2. **Aggression treated as cyberbullying.** Even the recent multilingual and Roman Urdu
   studies detect single-message *aggression* and equate it with cyberbullying, ignoring
   the **repetition** and **intent to harm** that the accepted definition requires.
3. **No large code-switched dataset and weak code-switching support.** There is a shortage
   of datasets combining Urdu, Roman Urdu and English, and existing models cope poorly with
   the mixed text that dominates South Asian social media.
4. **Text classified without behavioural context.** Few systems integrate the textual
   signal with repetition, intent and peerness, and fewer still do so within a single,
   coherent architecture.

This thesis addresses the gap directly by developing a **multilingual, two-stage,
context-aware framework** that (i) detects aggression across English and Roman Urdu using
multilingual transformers, and (ii) combines that signal with quantitative repetition,
intent and peerness features to make a relationship-level cyberbullying decision — thereby
modelling the full behavioural definition of cyberbullying rather than a single facet of
it.

**Table 2.1.** Summary of related work on cyberbullying and aggression detection.

| Theme | Representative work | Focus | Gap addressed by this thesis |
|---|---|---|---|
| Aggression-focused (Roman Urdu) | Dewani et al. (2021, 2023); Anwar & Anwar (2022); Rasheed et al. (2022) | Detect aggressive/abusive Roman Urdu comments with NLP and deep learning | Ignore repetition and intent; classify aggression, not full cyberbullying |
| Lack of multilingual datasets | Khan & Qureshi (2022); Akhter et al. (2020) | Highlight the need for Urdu / Roman Urdu / English code-switched data | Train on combined English + Roman Urdu data with multilingual transformers |
| Transformer models | Devlin et al. (BERT, 2018); Khanuja et al. (MuRIL, 2021); Das et al. (2022) | Transformer representations for low-resource languages | Applied to aggression only; extended here to repetition + intent |
| Defining cyberbullying | Stewart & Petermann (2018); Ejaz, Razi & Choudhury (2024) | Deliberate, repetitive, harmful behaviour; aggression + repetition + peerness + intent | Operationalised as a two-stage, relationship-level model |
| Multilingual detection | Razi & Ejaz (2024) | High accuracy on Urdu / Roman Urdu / English | Focused on aggression; adds repetition, intent, peerness |
| Deep-learning architectures | El Koshiry (2024) | Transformers outperform CNN/RNN for cyberbullying | Confirms transformer choice for Stage 1 |
| Behavioural/contextual cues | Ting et al.; Nurrahmi et al.; Hasan (2023) | Sentiment, user history, intent as features | Central to Stage 2 (repetition, intent, peerness) |

## 2.11 Summary

This chapter has traced the evolution of cyberbullying detection from keyword filtering and
classical machine learning, through deep learning, to multilingual transformer models, and
has reviewed the emerging work on Urdu and Roman Urdu. Two themes recur throughout. First,
the field is **strongly English-centric**, and the multilingual work that does exist must
contend with the unstandardised spelling and code-switching of South Asian social media.
Second, and more fundamentally, almost all systems — including the most recent multilingual
ones — **reduce cyberbullying to single-message aggression**, discarding the repetition and
intent that define it. The framework developed in the following chapter is designed to
overcome both limitations.
