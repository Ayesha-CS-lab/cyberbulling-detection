# CHAPTER 5

# CONCLUSION AND FUTURE WORK

## 5.1 Conclusion

This thesis set out to build and test a **context-aware, multilingual, two-stage framework for
cyberbullying detection** in code-switched Urdu, Roman Urdu and English. Two problems started it
off: most systems are built for **English only**, which leaves Urdu-speaking users unprotected;
and most **treat any single aggressive message as bullying**, throwing away the repetition and
intent that the real definition needs.

The framework answers both. **Stage 1** fine-tunes a multilingual transformer (m-BERT or MuRIL)
on 92,308 English and Roman Urdu messages to spot **aggression**, with class weighting to handle
the imbalance. **Stage 2** gathers that signal across each of 9,511 user-pairs and combines it with
measures of **repetition**, **intent to harm** and **peerness**, plus user context, to make the
final decision. In other words, the system reads cyberbullying as a **behaviour between two people
over time**, not as a single message.

The numbers bear this out. Stage 1 detects aggression reliably in both languages, with **MuRIL the
strongest** (88.2% accuracy, 0.840 F1, 0.889 recall), well ahead of the SVM and BiLSTM baselines
and matching the expected jump from classical methods to transformers. Stage 2 reaches **92.1%
accuracy with recall of 0.97–0.99**, catching 196 of 198 real bullying pairs and staying steady
across cross-validation folds. Just as important, the project caught and fixed a **majority-class
collapse** in an earlier model — one that showed a misleading 72% "accuracy" while detecting
nothing — which is a reminder that on imbalanced, safety-critical tasks the choice of metric and
framing matters as much as the model. A substantial inter-annotator agreement (Fleiss' κ = 0.67)
confirms the labels are trustworthy. A separate image-and-text extension based on a **frozen CLIP**
encoder was also built; on the Hateful Memes benchmark its fusion of image and text beat both
single-modality baselines, giving a working — if moderate — image capability alongside the text
system. The whole text pipeline is available through an interactive web app that classifies new
messages across all three languages in real time.

## 5.2 Key Contributions

The main contributions are:

1. A **multilingual, code-switching-aware aggression detector** on m-BERT / MuRIL, trained jointly
   on English and Roman Urdu, which pushes cyberbullying detection past its usual English-only
   setting.
2. A **two-stage framework** that separates message-level aggression from relationship-level
   cyberbullying, and so captures the repetition, intent and peerness that define real bullying
   rather than treating a single aggressive message as the whole story.
3. **Quantitative repetition and intent scoring** that works over a user-pair's history in both
   English and Roman Urdu.
4. An **honest, imbalance-aware evaluation**, including the diagnosis and fix of a majority-class
   collapse — a clear example of the difference between a model that looks like it works and one
   that does.
5. A **CLIP-based multimodal meme classifier** in which image-and-text fusion beats both unimodal
   baselines, extending the framework toward image-based cyberbullying.
6. A **web application** that shows real-time, multilingual detection.

## 5.3 Limitations

A few limitations should be acknowledged:

- **Language balance.** The message corpus is heavily English (only about 2% Roman Urdu). The
  transformers transfer well, but a larger native Roman Urdu and Urdu-script corpus would make the
  multilingual claim stronger.
- **Dataset scope.** The pair-level data comes from a bounded community of users; behaviour on open,
  platform-scale social media may differ.
- **Precision–recall trade-off.** Stage 2 is tuned for high recall, which lowers precision (≈ 0.57);
  flagged cases are best sent for human review rather than automatic action.
- **Intent scoring.** Intent is read from severity-weighted keyword and pattern cues, so it still
  misses sarcasm, irony and implicit threats — hard cases for any current system.
- **Image module performance.** The multimodal component (Section 4.7) works but is moderate: with a
  frozen CLIP encoder, fusion beats both unimodal baselines (AUC 0.651 against 0.641 image-only and
  0.606 text-only), yet its absolute score is well below the text system. It is a working component
  rather than a high-accuracy one, and multilingual (Urdu / Roman Urdu) memes are out of scope for
  lack of data.

## 5.4 Future Work

Several directions follow naturally:

1. **Strengthening image-based detection.** The image module was rebuilt on a **frozen CLIP encoder**
   with a light fusion head and **AUC-based selection** (Section 3.12), which cured the overfitting
   and collapse of the initial ResNet50 + m-BERT attempt and made fusion beat both single-modality
   baselines. Its absolute accuracy is still moderate, and a few things would push it further:
   - **A complete, larger, purpose-built dataset** — only a partial mirror of Hateful Memes was
     available; a full corpus labelled specifically for *bullying* (ideally sharing users with the
     conversational data, so real cross-modal fusion becomes possible) would help most.
   - **A bigger CLIP backbone with light fine-tuning** — moving to a larger CLIP model, and gently
     fine-tuning rather than fully freezing, would likely add several AUC points.
   - **Multilingual memes** — Urdu and Roman Urdu meme data is scarce; building such a set would let
     the image module match the multilingual text system.
   - **True multimodal fusion** — joining the image branch with the two-stage text model so text,
     image and behavioural signals all feed one decision.
2. **Richer Roman Urdu and Urdu-script data.** A bigger native Roman Urdu corpus, plus Urdu
   (Nastaʿlīq) script, would improve robustness and reduce the reliance on English.
3. **Context and sarcasm modelling.** Adding conversational context, sentiment and sarcasm detection
   would sharpen the intent estimate, which is currently keyword- and pattern-based.
4. **Real-time, platform-scale deployment.** Optimising for streaming and testing on live social
   media would show whether the pipeline holds up in practice.
5. **Explainability.** Attention-based explanations that highlight the words and features behind each
   decision would add transparency and trust, in line with the wider move toward explainable AI.

To close, this thesis has shown that detecting cyberbullying in multilingual, code-switched text is
not best done by judging isolated messages, but by modelling the aggression, repetition, intent and
social context of a relationship over time. The two-stage framework offers a culturally aware,
behaviourally complete and honestly evaluated foundation for safer online spaces in Urdu-speaking
communities — and a clear path toward multimodal and real-time extension.
