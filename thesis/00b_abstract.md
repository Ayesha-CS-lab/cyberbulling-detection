# ABSTRACT

Cyberbullying has become one of the most damaging consequences of the growth of social
media, with users exploiting online platforms to deliberately harm, threaten or humiliate
others. Its effects — anxiety, depression and social isolation — are well documented, yet
automated detection remains difficult in multilingual societies such as Pakistan, where
users routinely mix **Urdu, Roman Urdu and English** within a single conversation
(code-switching). Two limitations dominate the existing literature: most systems are built
for English text only, and most reduce cyberbullying to a single act of **aggression**,
ignoring the **repetition** and **intent to harm** that distinguish genuine bullying from
an isolated offensive remark.

This thesis presents a **context-aware, multilingual, two-stage deep-learning framework**
that detects cyberbullying in code-switched Urdu, Roman Urdu and English text. Rather than
classifying a single message in isolation, the framework reconstructs the bullying
behaviour at the level of a **user-to-user relationship**. In **Stage 1**, a multilingual
transformer (**m-BERT / MuRIL**) is fine-tuned on a combined English and Roman Urdu corpus
of **92,308 messages** to detect message-level **aggression**, with class weighting used to
counter the aggressive/non-aggressive imbalance. In **Stage 2**, the aggressive-message
signal is aggregated across the conversation history of each of **9,511 user-pairs** and
combined with quantitative measures of **repetition** (frequency and persistence of attacks
over time), **intent to harm** (severity-weighted linguistic cues in both English and Roman
Urdu) and **peerness** (the social relationship between the two users) to produce a final
cyberbullying decision through a compact dense neural network.

The framework is grounded in a comprehensive, pre-annotated cyberbullying dataset that
labels aggression, repetition, peerness and intent, and the reliability of the annotations
is assessed using **Fleiss' Kappa** (κ = 0.67, substantial agreement). Experimental results
show that the two-stage design recovers the full behavioural definition of cyberbullying
that single-stage aggression classifiers miss. Stage 1 detects multilingual aggression
reliably, with **MuRIL the strongest model (88.2% accuracy, 0.84 F1, 0.89 recall)**,
outperforming SVM and BiLSTM baselines, and Stage 2 classifies cyberbullying at the pair
level with **92.1% accuracy and a recall of 0.97–0.99**, correctly identifying **196 of 198**
true bullying relationships while remaining stable across cross-validation folds. The work
additionally documents and corrects a common failure mode — a **majority-class collapse**
that produced a misleadingly high 72% "accuracy" while detecting nothing — providing an
instructive contrast between a model that *appears* to work and one that genuinely does. A
**preliminary multimodal (image-and-text) component** for meme-based bullying was also
implemented and evaluated; on the Memotion benchmark it performed near the majority-class
baseline, and it is therefore reported as an exploratory result with the improvements needed
for reliable image-based detection identified as future work. The complete text system is
deployed as an interactive web demonstration that classifies new messages across all three
languages in real time.

The proposed framework offers a culturally aware, behaviourally complete and honestly
evaluated foundation for multilingual cyberbullying detection, supporting safer and more
respectful online environments for Urdu-speaking communities.

**Keywords:** Cyberbullying detection, multilingual NLP, Roman Urdu, code-switching,
deep learning, transformers, m-BERT, MuRIL, aggression detection, repetition, intent to
harm, peerness, Fleiss' Kappa.
