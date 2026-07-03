# CHAPTER 2

# LITERATURE REVIEW

## 2.1 Introduction

Work on detecting cyberbullying sits where three fields meet: natural language processing,
machine learning and online-safety research. Over the last ten years the methods have moved
on quickly — from simple word filters and classic classifiers, to deep networks, and lately
to large multilingual transformers. This chapter walks through the parts of that story that
matter here. It starts with how cyberbullying is actually defined, and why the definition
shapes the whole system. It then covers traditional machine-learning and lexicon methods, the
shift to deep learning and transformers, and the smaller but growing body of work on Urdu and
Roman Urdu. Finally it looks at code-switching, behavioural features, and the practical
questions around datasets and annotation, before naming the gap this thesis sets out to fill.

## 2.2 Defining Cyberbullying: A Behavioural View

Before building a detector, it helps to be clear about what is being detected. The most
widely quoted definition calls cyberbullying **an aggressive, intentional act carried out
through electronic means, repeatedly and over time, against someone who cannot easily defend
themselves** [3]. Stewart and Petermann (2018) [18] say much the same thing in different
words — deliberate, repeated, harmful use of technology — and stress the psychological motive
and the frequency of the abuse.

Two things follow. First, **a single offensive message is not, by itself, cyberbullying**;
repetition and intent are part of the definition, not optional extras. Second, cyberbullying
really describes a *relationship that plays out over time* between an aggressor and a victim,
not one isolated piece of text. Yet most detection work still boils the problem down to
labelling single messages as aggressive or not. This thesis instead follows the fuller
formulation of Ejaz, Razi and Choudhury (2024) [3], whose dataset and framework explicitly
build in **aggressive text, repetition, peerness and intent to harm** — the four ideas used
throughout this work.

## 2.3 Traditional and Machine-Learning Approaches

The earliest attempts at spotting abusive language leaned on **classic supervised learning**.
The recipe was familiar: collect and label some data, clean the text, turn it into features
(usually bag-of-words or n-gram counts, sometimes weighted by TF–IDF), and train a classifier
such as an SVM, Naïve Bayes, Logistic Regression or SGD.

Several studies show both the range and the limits of this approach. Working on Arabic tweets,
Haidar et al. [29] used SVM and Naïve Bayes and reported F1 scores around 0.92 and 0.90.
Özel et al. [30] classified Turkish posts from Twitter and Instagram with decision trees, SVM,
Multinomial Naïve Bayes and k-Nearest Neighbours, and found that adding emoticons as features
helped, with Naïve Bayes coming out on top near 84%. Nurrahmi et al. [31] tackled Indonesian
with an SVM and also tracked each user's history to gauge credibility, reaching an F1 of about
0.67. In a multilingual system spanning English, Hindi and Marathi, Pawar (2019) [36] combined
Multinomial Naïve Bayes, SGD and Logistic Regression inside a distributed architecture and
found that machine-learning models beat lexicon-based ones in every language.

These works set important baselines, but they share two weaknesses. They usually treat text as
a **bag of separate words**, so word order, grammar and negation are lost. And nearly all of
them are monolingual, and mostly English — performance falls apart on the informal, misspelt,
code-switched text of real social media.

## 2.4 Lexicon-Based Approaches

A second line of work is **lexicon-based**: keep a dictionary of abusive or offensive terms,
and flag bullying by matching messages against it. Chen et al. [33] proposed the **Lexical
Syntactic Feature (LSF)** method for sentence-level offensive-language detection and reported
high precision and recall on English. Kontostathis et al. [34] studied the specific vocabulary
cyberbullies use and built queries from those terms to pull out bullying content.

Lexicon methods are transparent and need no training data, but they are brittle. They only
know the words in the list; they cannot read novel slang, sarcasm or deliberately misspelt
words; and one unrecognised word can flip a whole message's score. That fragility is worst for
**Roman Urdu**, which has no standard spelling — a single abusive word can appear in many
forms, and no fixed dictionary will hold them all. For that reason this thesis uses lexicon
matching only as a supporting signal (inside the intent scorer), never as the main classifier.

## 2.5 Deep Learning and Transformer Models (BERT, m-BERT, MuRIL)

Hand-built features could only go so far, which pushed the field toward **deep learning**,
where the features are learned from data. CNNs, RNNs and LSTMs improved on the classic
baselines by picking up local patterns and word order. Khan et al. (2022) [9], for instance,
applied a CNN–LSTM to English and Roman Urdu sentiment data, showing that deep models could be
trained on Roman Urdu directly.

The real turning point was the **Transformer**, and especially **BERT** [26]. By pre-training
a bidirectional language model on huge amounts of text and then fine-tuning it, BERT captures
deep, context-dependent meaning that earlier models missed. El Koshiry (2024) [24] found that
transformers beat older deep-learning models like CNNs and RNNs on cyberbullying tasks.

For multilingual settings, two transformer variants matter most here, and both are used in
this thesis:

- **Multilingual BERT (m-BERT)** is pre-trained on Wikipedia text in over a hundred languages,
  so it learns a shared space where words from different languages relate to one another — a
  natural fit for code-switched text.
- **MuRIL** [27] is pre-trained specifically on South Asian languages and, importantly, on
  **transliterated** text, which is exactly what Roman Urdu is.

The catch is that almost all of this work applies transformers to **aggression or hate-speech
classification of single messages**. They are rarely, if ever, used to capture the
*repetition* and *intent* that the definition of cyberbullying demands — a gap between what the
models can read and what the behaviour actually is. Closing that gap is the point of this
thesis.

## 2.6 Cyberbullying and Aggression Detection in Roman Urdu

A focused stream of work now targets Urdu and Roman Urdu directly. Dewani et al. (2021, 2023)
[13], Anwar and Anwar (2022) [14] and Rasheed et al. (2022) [15] all built NLP and
deep-learning models for detecting **aggressive Roman Urdu comments**, with strong accuracy.
Razi and Ejaz (2024) [19] went further with a multilingual system covering Urdu, Roman Urdu
and English, again reporting high accuracy on a multilingual set.

These studies are valuable, and they prove that Roman Urdu abuse *can* be detected. But they
all share the same limit: they treat cyberbullying as aggression. Each decides whether one
comment is aggressive, and none brings in repetition or intent. So they cover just one of the
three behavioural pillars from Section 2.2, and cannot tell a one-off insult apart from a
sustained campaign.

## 2.7 The Code-Switching Problem in South Asian Social Media

Several authors point to **code-switching** — mixing languages within a single message — as a
central, unsolved difficulty. Khan and Qureshi (2022) [16] and Akhter et al. (2020) [17] note
that real posts routinely blend Urdu, Roman Urdu and English, while the datasets available for
training rarely do; most are stuck in a single language, which caps how well any model trained
on them can generalise. They single out the **creation of genuinely multilingual, code-switched
datasets** as a key requirement for progress.

Two consequences follow for system design. Monolingual models and fixed word lists are simply
the wrong tool for mixed input. And a model has to *see* the same kind of mixed text during
training that it will meet at test time. This thesis answers both by training its aggression
model on a combined English and Roman Urdu corpus with transformers that share meaning across
languages.

## 2.8 Behavioural and Contextual Features

A smaller set of studies recognises that text alone is not enough and adds **behavioural
context**. Ting et al. [32] mined social-network features — keywords, network structure and
sentiment — and found sentiment, as a stand-in for intent, to be the single most useful one.
Nurrahmi et al. [31] tracked each user's ratio of bullying to non-bullying messages to judge
how credible a potential aggressor was. Silva et al. [35], in *BullyBlocker*, kept a 60-day
record of a user's activity rather than judging messages one at a time.

These point straight at the features this thesis puts at the centre: **repetition** (how often
and how persistently aggression recurs), **intent** (how severe and purposeful the language
is) and **peerness** (the relationship and power balance between two users). Stewart and
Petermann's [18] focus on repetition and motive, together with Mahmud et al.'s (2023) [37]
call for *intent detection* in identifying real cyberbullying, both reinforce the case for
combining the text signal with these relational features instead of relying on text alone.

## 2.9 Datasets, Annotation and Inter-Annotator Agreement

Progress in this area is limited by how much good labelled data exists. Mahmud et al. (2023)
[37] flag the shortage of datasets for low-resource languages such as Urdu and Hindi as a
basic obstacle, and note that a lot of the work stays theoretical for want of a suitable
corpus. The dataset used in this thesis, drawn from Ejaz, Razi and Choudhury (2024) [3], stands
out precisely because it labels not only aggression but also repetition, peerness and intent at
the level of user relationships.

Where humans assign labels, their **reliability** has to be shown, not assumed. The standard
tool for measuring agreement among several annotators on categorical labels is **Fleiss'
Kappa**, which corrects the observed agreement for what you would expect by chance. Reporting
it is how a dataset earns trust, so this study measures agreement with Fleiss' Kappa
(Section 3.4).

## 2.10 Research Gap

Put together, the literature shows a consistent, layered gap, summarised in **Table 2.1**:

1. **English dominance.** Most research targets English, leaving Urdu and Roman Urdu users
   underserved.
2. **Aggression treated as cyberbullying.** Even recent multilingual and Roman Urdu work
   detects single-message *aggression* and calls it bullying, ignoring the **repetition** and
   **intent** the definition requires.
3. **Weak support for code-switching.** There is a shortage of Urdu / Roman Urdu / English
   datasets, and existing models cope badly with the mixed text of South Asian platforms.
4. **Text without behavioural context.** Few systems combine the text with repetition, intent
   and peerness, and fewer still do it in one coherent design.

This thesis tackles the gap head-on with a **multilingual, two-stage, context-aware framework**
that (i) detects aggression across English and Roman Urdu with multilingual transformers, and
(ii) combines that signal with quantitative repetition, intent and peerness to make a
relationship-level decision — modelling the whole behaviour of cyberbullying rather than one
slice of it.

**Table 2.1.** Summary of related work on cyberbullying and aggression detection.

| Theme | Representative work | Focus | Gap addressed by this thesis |
|---|---|---|---|
| Aggression-focused (Roman Urdu) | Dewani et al. (2021, 2023); Anwar & Anwar (2022); Rasheed et al. (2022) | Detect aggressive/abusive Roman Urdu comments with NLP and deep learning | Ignore repetition and intent; classify aggression, not full cyberbullying |
| Lack of multilingual datasets | Khan & Qureshi (2022); Akhter et al. (2020) | Highlight the need for Urdu / Roman Urdu / English code-switched data | Train on combined English + Roman Urdu data with multilingual transformers |
| Transformer models | Devlin et al. (BERT, 2018); Khanuja et al. (MuRIL, 2021) | Transformer representations for low-resource languages | Applied to aggression only; extended here to repetition + intent |
| Defining cyberbullying | Stewart & Petermann (2018); Ejaz, Razi & Choudhury (2024) | Deliberate, repetitive, harmful behaviour; aggression + repetition + peerness + intent | Operationalised as a two-stage, relationship-level model |
| Multilingual detection | Razi & Ejaz (2024) | High accuracy on Urdu / Roman Urdu / English | Focused on aggression; adds repetition, intent, peerness |
| Deep-learning architectures | El Koshiry (2024) | Transformers outperform CNN/RNN for cyberbullying | Confirms transformer choice for Stage 1 |
| Behavioural/contextual cues | Ting et al.; Nurrahmi et al.; Mahmud et al. (2023) | Sentiment, user history, intent as features | Central to Stage 2 (repetition, intent, peerness) |

## 2.11 Summary

This chapter traced cyberbullying detection from word filters and classic machine learning,
through deep learning, to multilingual transformers, and reviewed the newer work on Urdu and
Roman Urdu. Two themes kept returning. The field is **heavily English-centric**, and the
multilingual work that does exist still has to wrestle with the unstandardised spelling and
code-switching of South Asian social media. More fundamentally, almost every system — even the
newest multilingual ones — **reduces cyberbullying to single-message aggression**, throwing
away the repetition and intent that make it what it is. The framework in the next chapter is
designed to fix both problems.
