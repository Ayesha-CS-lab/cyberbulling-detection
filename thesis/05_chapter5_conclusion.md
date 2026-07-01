# CHAPTER 5

# CONCLUSION AND FUTURE WORK

## 5.1 Conclusion

This thesis presented the design, implementation and evaluation of a **context-aware,
multilingual, two-stage deep-learning framework for cyberbullying detection** in
code-switched Urdu, Roman Urdu and English text. The work was motivated by two
persistent limitations in the existing literature: the dominance of **English-only**
systems, which leave Urdu-speaking online communities unprotected, and the widespread
practice of **equating cyberbullying with single-message aggression**, which discards
the repetition and intent that the accepted definition of cyberbullying requires.

The proposed framework addresses both limitations through a coherent design. **Stage 1**
fine-tunes a multilingual transformer (m-BERT or MuRIL) on a combined corpus of 92,308
English and Roman Urdu messages to detect message-level **aggression**, using class
weighting to counter the aggressive/non-aggressive imbalance. **Stage 2** aggregates this
aggression signal across the conversation history of each of 9,511 user-pairs and combines
it with quantitative measures of **repetition**, **intent to harm** and **peerness**,
together with user context, to make a final, relationship-level cyberbullying decision
through a compact dense network. In doing so, the framework models cyberbullying as a
**behaviour over a relationship**, not as a property of an isolated message.

The experimental results confirm the value of this design. Stage 1 detects aggression
reliably across both languages, with **MuRIL the strongest model** (88.2% accuracy, 0.840
F1 and 0.889 recall), comfortably outperforming the SVM and BiLSTM baselines and confirming
the expected progression from classical machine learning to transformers. Stage 2 classifies
cyberbullying at the relationship level with **92.1% accuracy and a recall of 0.97–0.99**,
correctly identifying 196 of 198 true cyberbullying pairs while remaining stable across
cross-validation folds. Equally important, the project diagnosed and corrected a
**majority-class collapse** in an earlier model — which had produced a misleading 72%
"accuracy" while detecting nothing — demonstrating that on imbalanced, safety-critical tasks
the choice of metric and problem framing matters as much as the choice of model. The
reliability of the underlying annotations was confirmed by a substantial inter-annotator
agreement (Fleiss' κ = 0.67). The complete system was made accessible through an interactive
web demonstration that classifies new messages across all three languages in real time.

## 5.2 Key Contributions

The principal contributions of this thesis are:

1. **A multilingual, code-switching-aware aggression detector** built on m-BERT / MuRIL and
   trained jointly on English and Roman Urdu, extending cyberbullying detection beyond its
   usual English-only setting.
2. **A two-stage, behaviourally complete framework** that separates message-level aggression
   from relationship-level cyberbullying, modelling the repetition, intent and peerness that
   define genuine bullying rather than equating it with a single aggressive message.
3. **Quantitative repetition and intent scoring** that operate over a user-pair's conversation
   history and function in both English and Roman Urdu.
4. **An honest, imbalance-aware evaluation**, including the explicit diagnosis and correction
   of a majority-class-collapse failure — a documented contrast between a model that *appears*
   to work and one that genuinely does.
5. **A deployable web application** that demonstrates real-time, multilingual cyberbullying
   detection.

## 5.3 Limitations

Several limitations are acknowledged:

- **Language balance.** The message-level corpus is strongly English-dominant (only about
  2% Roman Urdu). While the multilingual transformers transfer well, a larger native Roman
  Urdu and Urdu-script corpus would strengthen the multilingual claim.
- **Dataset scope.** The relationship-level data is drawn from the conversation history of a
  bounded community of users; performance on open, platform-scale social media may differ.
- **Precision–recall trade-off.** Stage 2 is deliberately tuned for high recall, which lowers
  precision (≈ 0.57); flagged cases are therefore best suited to human review rather than
  fully automatic action.
- **Intent scoring.** Intent is estimated from severity-weighted keyword and pattern cues; it
  does not yet capture sarcasm, irony or implicit threats, which remain difficult for any
  current system.
- **Separate modalities.** The image (meme) component is evaluated on a separate dataset from
  the conversational text system, as the two data sources do not share users.
- **Preliminary image performance.** The multimodal image component (Section 4.7) did not
  reach a reliable level: on the Memotion benchmark it performed near the majority-class
  baseline (macro-F1 ≈ 0.51) and showed overfitting and majority-class collapse. The image
  modality is therefore reported as exploratory rather than as a validated contribution.

## 5.4 Future Work

A number of directions follow naturally from this work:

1. **Improving image-based detection.** The preliminary multimodal experiment (Section 4.7)
   exposed several concrete obstacles to reliable image-based cyberbullying detection, each
   of which defines a direction for future work:
   - **A purpose-built bullying image dataset.** The Memotion "offensive" label is only a weak
     proxy for cyberbullying; a larger dataset annotated specifically for *bullying* (rather
     than general offensiveness or sentiment) is needed, ideally one that shares users with
     the conversational data so true multimodal fusion is possible.
   - **Avoiding overfitting on small data.** Fully fine-tuning two large backbones (ResNet50
     and m-BERT) on a few thousand memes overfits badly; freezing the pretrained backbones,
     using lightweight fusion, and applying data augmentation are needed for stable training.
   - **Handling class imbalance honestly.** The component drifted toward majority-class
     prediction; balanced sampling, threshold tuning and **macro-averaged** model selection
     (rather than selecting on a single class's F1) are required to prevent the collapse seen
     in Section 4.6.
   - **Better text-in-image handling.** Because the bullying cue in a meme lies largely in its
     text, higher-quality OCR and stronger text–image alignment (for example CLIP-style
     encoders) would likely help more than a larger image backbone alone.
   - **True multimodal fusion.** Once reliable, the image branch can be fused with the
     conversational two-stage model so that text, image and behavioural signals contribute to
     a single cyberbullying decision.
2. **Richer Roman Urdu and Urdu-script data.** Expanding the native Roman Urdu corpus and
   adding Urdu (Nastaʿlīq) script would improve multilingual robustness and reduce reliance on
   the English-dominant majority.
3. **Context and sarcasm modelling.** Incorporating conversational context, sentiment and
   sarcasm detection would refine the intent estimate, which is currently keyword- and
   pattern-based.
4. **Real-time, platform-scale deployment.** Optimising the pipeline for streaming
   classification and evaluating it on live social-media data would test its operational
   viability.
5. **Explainability.** Adding attention-based explanations that highlight the words and
   behavioural features driving each decision would increase transparency and trust, mirroring
   the move toward explainable AI in related domains.

In conclusion, this thesis has shown that detecting cyberbullying in multilingual,
code-switched text is best achieved not by classifying isolated messages, but by modelling the
aggression, repetition, intent and social context of a relationship over time. The proposed
two-stage framework provides a culturally aware, behaviourally complete and honestly evaluated
foundation for safer online environments in Urdu-speaking communities, and a clear path toward
multimodal and real-time extension.
