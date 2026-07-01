# INSERT — Multimodal Image+Text Component

> **How to use this file.** This is ready-to-slot thesis text for the image
> (multimodal) component. Add Section 3.12 to the end of Chapter 3, and Section
> 4.x to Chapter 4 after the Stage 2 results. Fill the **[bracketed]** numbers
> from your Kaggle run (`models/evaluation_report_image.txt`). If you keep the
> image component, also update the title (remove the "future work" note) and the
> abstract to mention the multimodal classifier.

---

## 3.12 Stage 3 — Multimodal Image-and-Text Detection

Cyberbullying online is not confined to plain text: a large share of abusive
content takes the form of **memes**, in which an image and a short caption combine
to convey a hostile or humiliating message. To address this modality — and to
fulfil the image-based dimension of the study — a third, **multimodal** classifier
is developed in which each sample consists of an image together with its overlaid
text.

The architecture (`fusion_model.py`) encodes the two modalities with complementary
backbones and combines them through attention:

- **Image branch.** A **ResNet50** convolutional network, pretrained on ImageNet,
  encodes the meme image into a 2048-dimensional visual feature vector.
- **Text branch.** The same multilingual transformer used in Stage 1
  (**m-BERT / MuRIL**) encodes the meme's text into a 768-dimensional embedding.
- **Attention fusion.** Each modality is projected to a common 512-dimensional
  space, and a learned **attention layer** computes a weighted combination of the
  two, allowing the model to rely on whichever modality is more informative for a
  given sample. A dense classification head then predicts **bullying / not
  bullying**.

The model is trained on the **[dataset name]** dataset of **[N]** labelled memes
(**[P]** bullying / **[Q]** non-bullying), split into stratified training and test
sets. Because abusive memes are the minority class, the same **positive-class
weighting** used in the text stages is applied here, and the model is again
evaluated using precision, recall and F1-score rather than accuracy alone. This
multimodal classifier is a **standalone component** evaluated on meme data; it
complements, rather than replaces, the two-stage conversational text system, which
remains the core of the framework.

> **Figure 3.7.** Stage 3 multimodal architecture: ResNet50 (image) and m-BERT /
> MuRIL (text) features fused by an attention layer for a binary bullying decision.
> *(Figure to be inserted.)*

---

## 4.x Multimodal Image-and-Text Results

The multimodal classifier was trained and evaluated on the **[dataset name]**
dataset. **Table 4.x** reports its performance. The fusion model achieved an
accuracy of **[acc]**, precision of **[P]**, recall of **[R]** and F1-score of
**[F1]**, demonstrating that the attention-based combination of visual and textual
features can identify bullying memes that neither modality reliably captures on its
own. The attention weights additionally indicate the relative contribution of the
image and text channels to each decision.

**Table 4.x.** Multimodal image-and-text cyberbullying detection results.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| ResNet50 + m-BERT (fusion) | [ ] | [ ] | [ ] | [ ] |
| ResNet50 + MuRIL (fusion) | [ ] | [ ] | [ ] | [ ] |

Together with the two-stage text system, this component gives the framework
coverage of **both** of the channels through which cyberbullying spreads on social
media — conversational text and image-based memes — across Urdu, Roman Urdu and
English.
