# CHAPTER 1

# INTRODUCTION

## 1.1 Background

Social media changed how we talk to each other. In a couple of decades, platforms like
Facebook, Instagram, WhatsApp, Twitter (X) and YouTube went from novelties to the main way
billions of people stay in touch. That reach brought real benefits, but it also opened new
doors for harm. One of the worst is **cyberbullying** — using electronic messages to hurt,
threaten or humiliate someone, again and again.

Traditional bullying stops at the school gate. Cyberbullying does not. It follows the
victim home, runs day and night, reaches a large audience in seconds, and often hides behind
anonymous accounts so the attacker never has to face the person they are hurting. A cruel
message, once posted, is hard to erase and can resurface later. Because of this, the damage
tends to build up over time and cut deeper than an offline insult would.

The scale has grown with the platforms themselves. Large numbers of young people say they
have either been targeted or watched it happen to someone else, and a real share of online
activity involves abuse of some kind. No team of human moderators can read every message, so
there is a clear need for systems that can **flag cyberbullying automatically**, and at
scale.

This thesis works on a part of that problem that has been largely overlooked: the mixed,
**code-switched** text that is normal on South Asian — and especially Pakistani — social
media, where Urdu, Roman Urdu and English are thrown together in a single chat.

## 1.2 Cyberbullying and Its Effects

Researchers do not define cyberbullying as a single nasty comment. The common description is
**an aggressive, intentional act, repeated over time, carried out through electronic means
against someone who cannot easily defend themselves**. Three ideas show up again and again:
the behaviour is *harmful*, it is *repeated*, and it is *deliberate*. A rude remark thrown
out in the heat of an argument is unpleasant, but on its own it is not bullying. The same
remark aimed at the same person over and over, clearly meant to wound, is.

The effects on victims are serious. Depression, anxiety, fear, low self-esteem and a sense
of isolation are all common, and they often come with disturbed sleep, poor appetite, falling
grades and withdrawal from school. In the worst cases, sustained cyberbullying has been
linked to self-harm. Because the harm accumulates across many incidents rather than one, it
matters that we can spot an *ongoing* pattern early, not just a single bad message.

Platforms have added reporting buttons, blocking, muting and safety pages. Useful as these
are, almost all of them are **reactive**: they kick in only after the abuse is already posted,
and usually only once a human reports it. There is a strong case for catching harmful
behaviour as it unfolds instead.

## 1.3 The Multilingual Challenge: Urdu, Roman Urdu and Code-Switching

Nearly all of the research on automatic cyberbullying detection has been done in **English**.
That leaves out the millions of people who argue, joke and, sometimes, bully in other
languages. The gap is widest in South Asia, where social media is enormous and the way people
write online does not look like clean English at all.

In Pakistan and nearby regions, two things happen at once:

- **Roman Urdu** — Urdu written in the English alphabet rather than the Urdu script (for
  example *"tum bohat buray ho"* instead of "تم بہت برے ہو"). It has no fixed spelling, so the
  same word can be written a dozen ways, which makes dictionary-based methods almost useless.
- **Code-switching** — mixing two or more languages in one sentence (for example
  *"yaar tum to total loser ho, kabhi sudhroge nahi"*).

A model trained only on English cannot read Roman Urdu, a single-language model loses the
thread when the language flips mid-sentence, and a list of banned words cannot keep up with
slang that changes weekly. In short, this problem needs models that are **multilingual from
the start**. Recent transformer models — in particular **Multilingual BERT (m-BERT)** and
**MuRIL**, which was trained on South Asian and transliterated text — fit that requirement,
because they learn a shared sense of meaning across languages.

## 1.4 Beyond Aggression: Repetition, Intent and Peerness

There is a second gap, and it is about *what* systems actually detect. Most tools — including
most of the work on Roman Urdu — reduce the task to **aggression detection**: given one
message, decide if it is abusive. That is a useful piece, but it is not cyberbullying. As
Section 1.2 argued, bullying is defined by **repetition** and **intent** as much as by
aggression. A system that labels every aggressive message as "cyberbullying" will keep
mislabelling one-off insults and jokes between friends, and it will never recognise a slow,
escalating campaign of harassment for what it is.

This thesis follows a fuller, behavioural view built around four ideas:

- **Aggression** — is a single message hostile or abusive?
- **Repetition** — do aggressive messages from the same sender to the same target keep coming
  back over time?
- **Intent to harm** — does the language show a real purpose to threaten, degrade or
  intimidate?
- **Peerness** — what is the relationship and power balance between the two users (for example
  their age or grade), which helps tell bullying apart from banter between equals?

Put simply, aggression is the raw signal in one message, while repetition and intent — read in
the light of peerness — are what turn a run of aggressive messages into cyberbullying. The
framework in this thesis is built directly around that distinction.

## 1.5 Problem Statement

Deep-learning methods for online abuse have come a long way, but several gaps still limit
their use on real, multilingual social media:

1. **They are built for English.** Most systems do not transfer to Urdu or Roman Urdu, so a
   large population of users is left unprotected.
2. **They cannot handle code-switching.** Large labelled datasets that mix Urdu, Roman Urdu
   and English are scarce, and most models cannot process the mixed text that dominates South
   Asian platforms.
3. **They equate aggression with cyberbullying.** Classifying a single message as aggressive
   is treated as the whole job, so isolated remarks get flagged as bullying while genuinely
   repeated, targeted harassment goes unrecognised.
4. **They ignore behavioural context.** Few approaches combine the text with how often the
   aggression recurs, how harmful the intent is, and how the two users relate — all of which
   are needed to make the right call.

This thesis addresses these gaps with a **multilingual, context-aware, two-stage framework**
that detects aggression across Urdu, Roman Urdu and English, and then combines it with
repetition, intent and peerness to reach a relationship-level cyberbullying decision.

## 1.6 Motivation

The motivation is partly social and partly technical. **Socially**, the users hit hardest by
the English-only bias of current tools are exactly those in multilingual, lower-resource
communities — Urdu speakers among them — who are very active online yet least protected.
Building a detector that understands Roman Urdu and mixed text is a step towards fairer
coverage.

**Technically**, the project grew out of a hard lesson about misleading accuracy. In real
data, bullying messages are far rarer than ordinary ones. A lazy model can score a high
"accuracy" simply by guessing that nothing is ever bullying — while catching none of it. An
early version of this system did exactly that: it reported a respectable-looking accuracy but
failed to find a single real case (Section 4.6). Fixing that failure shaped the main design
choices here — class weighting, the use of precision, recall and F1 instead of accuracy alone,
and a two-stage design that models behaviour rather than the surface of one message.

## 1.7 Aim and Objectives

The **aim** of this project is to design, build and evaluate a **context-aware, multilingual
framework that detects cyberbullying in code-switched Urdu, Roman Urdu and English text** by
modelling aggression, repetition and intent within the social context of a relationship.

The **objectives** are to:

1. Assemble and preprocess a multilingual dataset (Urdu, Roman Urdu, English) and organise it
   into a message-level view for aggression and a user-pair view for cyberbullying.
2. Check the reliability of the labels for aggression, repetition, intent and peerness, using
   **Fleiss' Kappa** for inter-annotator agreement.
3. Fine-tune **m-BERT** and **MuRIL** for message-level aggression, using class weighting to
   handle imbalance.
4. Design quantitative scores for repetition (how persistent the aggression is) and intent to
   harm (severity-weighted cues in both English and Roman Urdu).
5. Build a second-stage classifier that fuses the aggression signal with repetition, intent,
   peerness and user context into a final decision.
6. Evaluate with accuracy, precision, recall and F1; compare m-BERT and MuRIL against SVM and
   LSTM baselines; and document the majority-class-collapse failure and its fix.
7. Deploy the system as an interactive web app for real-time, multilingual detection.

## 1.8 Research Questions

- **RQ1.** Can a multilingual transformer detect message-level aggression reliably across
  English and Roman Urdu, including code-switched text — and which model does it better?
- **RQ2.** Does splitting the task into message-level aggression and relationship-level
  cyberbullying recover the repetition and intent that single-stage models miss?
- **RQ3.** Can repetition and intent be measured from conversation history in a way that
  actually improves the final decision?
- **RQ4.** How does the corrected, class-balanced model compare with the earlier one that
  collapsed to a single class, and what does that tell us about evaluating imbalanced tasks?

## 1.9 Scope of the Study

This work focuses on **text-based** detection in three languages — English, Roman Urdu and
Urdu — within the user-to-user conversational setting of the dataset described in Chapter 3.
It covers the full pipeline: data preparation, label-reliability checks, multilingual
aggression detection, repetition and intent scoring, relationship-level classification,
comparison against baselines, and a web demo.

The core of the study is deliberately **text-based**. Although the proposal also imagined an
image side, the multilingual, code-switched behaviour at the heart of this problem lives in
text, and that is where the labelled data for aggression, repetition, intent and peerness
exists. A separate image-and-text (meme) component was built as an additional module
(Sections 3.12 and 4.7). Sarcasm detection, full sentiment and context modelling, and
platform-scale deployment are left for future work.

## 1.10 Significance and Contributions

The main contributions of this thesis are:

1. A **multilingual, code-switching-aware aggression detector** built on m-BERT / MuRIL and
   trained jointly on English and Roman Urdu.
2. A **two-stage framework** that separates message-level aggression from relationship-level
   cyberbullying, and so captures the repetition, intent and peerness that define real
   bullying.
3. **Quantitative repetition and intent scoring** that work over a user-pair's history in both
   English and Roman Urdu.
4. An **honest, imbalance-aware evaluation**, including the diagnosis and correction of a
   majority-class collapse — a clear example of a model that looks like it works but does not.
5. A **web application** that demonstrates real-time, multilingual detection.

## 1.11 Organisation of the Thesis

The rest of the thesis is organised as follows. **Chapter 2** reviews the relevant work and
sets out the research gap. **Chapter 3** describes the dataset, preprocessing, the annotation
and agreement check, the two stages, the scoring methods, the evaluation setup and the web
demo. **Chapter 4** presents and discusses the results, including the majority-class-collapse
story and the multimodal component. **Chapter 5** concludes, notes the limitations, and points
to future work.
