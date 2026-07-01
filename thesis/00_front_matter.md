# Front Matter — Title, Table of Contents, List of Figures, List of Tables

> **Working note:** Page numbers below are placeholders/estimates. They will be
> finalised automatically once the full chapter content is written and the thesis
> is compiled to Word (`.docx`), where the Table of Contents, List of Figures, and
> List of Tables are generated from the document's headings and captions.

---

# A CONTEXT-AWARE MULTILINGUAL DEEP LEARNING FRAMEWORK FOR CYBERBULLYING DETECTION IN CODE-SWITCHED URDU, ROMAN URDU AND ENGLISH TEXT

**Student Name:** Rimsha Ch
**Roll No:** 24252101017
**Supervisor:** Mam Fatima Anjum

Committee Member 1: ____________________
Committee Member 2: ____________________
Committee Member 3: ____________________

**DEPARTMENT OF COMPUTER SCIENCE**
**LAHORE COLLEGE FOR WOMEN UNIVERSITY, LAHORE**

> **Note on the title.** The original proposal was titled *"A Context-Aware
> Multilingual Deep Learning Model for Text and Image-Based Cyberbullying
> Detection."* The core of this study is the **text modality**, which is where
> multilingual code-switched cyberbullying actually occurs and where the comprehensive
> labelled data (aggression, repetition, intent and peerness) exists. An image-based
> **multimodal** component was also implemented and evaluated, but only as a
> **preliminary, exploratory** result (Sections 3.12 and 4.7); the improvements needed
> to make it reliable are set out as future work (Section 5.4). The title above reflects
> the validated core of the system that was actually built and evaluated.

---

# TABLE OF CONTENTS

| | | Page |
|---|---|---:|
| | **Declaration** | i |
| | **Certificate of Approval** | ii |
| | **Dedication** | iii |
| | **Acknowledgements** | iv |
| | **Abstract** | v |
| | **Table of Contents** | vi |
| | **List of Figures** | ix |
| | **List of Tables** | x |
| | **List of Abbreviations** | xi |
| | | |
| **CHAPTER 1** | **INTRODUCTION** | 1 |
| 1.1 | Background | 1 |
| 1.2 | Cyberbullying and Its Effects | 2 |
| 1.3 | The Multilingual Challenge: Urdu, Roman Urdu and Code-Switching | 3 |
| 1.4 | Beyond Aggression: Repetition, Intent and Peerness | 4 |
| 1.5 | Problem Statement | 5 |
| 1.6 | Motivation | 6 |
| 1.7 | Aim and Objectives | 7 |
| 1.8 | Research Questions | 8 |
| 1.9 | Scope of the Study | 8 |
| 1.10 | Significance and Contributions | 9 |
| 1.11 | Organisation of the Thesis | 10 |
| | | |
| **CHAPTER 2** | **LITERATURE REVIEW** | 11 |
| 2.1 | Introduction | 11 |
| 2.2 | Defining Cyberbullying: A Behavioural View | 12 |
| 2.3 | Traditional and Machine-Learning Approaches | 13 |
| 2.4 | Lexicon-Based Approaches | 14 |
| 2.5 | Deep Learning and Transformer Models (BERT, m-BERT, MuRIL) | 15 |
| 2.6 | Cyberbullying and Aggression Detection in Roman Urdu | 16 |
| 2.7 | The Code-Switching Problem in South Asian Social Media | 17 |
| 2.8 | Behavioural and Contextual Features | 18 |
| 2.9 | Datasets, Annotation and Inter-Annotator Agreement | 19 |
| 2.10 | Research Gap | 20 |
| 2.11 | Summary | 21 |
| | | |
| **CHAPTER 3** | **MATERIALS AND METHODS** | 22 |
| 3.1 | Overview of the Proposed Methodology | 22 |
| 3.2 | Dataset | 24 |
| 3.2.1 | The Comprehensive Cyberbullying Dataset | 24 |
| 3.2.2 | Roman Urdu Aggression Data | 25 |
| 3.2.3 | Message-Level and Pair-Level Views | 26 |
| 3.3 | Data Preprocessing | 27 |
| 3.3.1 | Text Cleaning and Normalisation | 27 |
| 3.3.2 | Roman Urdu Handling and Tokenisation | 28 |
| 3.4 | Data Annotation and Inter-Annotator Agreement | 29 |
| 3.4.1 | Annotation Scheme (Aggression, Repetition, Intent, Peerness) | 29 |
| 3.4.2 | Fleiss' Kappa Agreement | 30 |
| 3.5 | Feature Extraction | 31 |
| 3.5.1 | Textual Features (m-BERT / MuRIL Embeddings) | 31 |
| 3.5.2 | Behavioural and Contextual Features | 32 |
| 3.6 | Stage 1 — Message-Level Aggression Model | 33 |
| 3.6.1 | Transformer Fine-Tuning | 33 |
| 3.6.2 | Class Imbalance Handling | 34 |
| 3.7 | Quantifying Repetition and Intent | 35 |
| 3.7.1 | Repetition Scoring | 35 |
| 3.7.2 | Intent-to-Harm Scoring | 36 |
| 3.8 | Stage 2 — User-Pair Cyberbullying Classifier | 37 |
| 3.9 | Evaluation Metrics | 38 |
| 3.10 | Experimental Setup and Implementation Details | 39 |
| 3.11 | Web Demonstration Application | 40 |
| 3.12 | Stage 3 — Multimodal Image-and-Text Component | 40 |
| | | |
| **CHAPTER 4** | **RESULTS AND DISCUSSION** | 41 |
| 4.1 | Introduction | 41 |
| 4.2 | Dataset Statistics and Class Distribution | 41 |
| 4.3 | Inter-Annotator Agreement Results | 42 |
| 4.4 | Stage 1 — Aggression Detection Results | 43 |
| 4.5 | Stage 2 — Cyberbullying Classification Results | 45 |
| 4.6 | The "Before" Story: Diagnosing Majority-Class Collapse | 46 |
| 4.7 | Multimodal Image-and-Text Component (Stage 3) | 47 |
| 4.8 | Discussion | 48 |
| 4.9 | Comparison with Existing Literature | 49 |
| | | |
| **CHAPTER 5** | **CONCLUSION AND FUTURE WORK** | 51 |
| 5.1 | Conclusion | 51 |
| 5.2 | Key Contributions | 52 |
| 5.3 | Limitations | 53 |
| 5.4 | Future Work | 54 |
| | | |
| | **REFERENCES** | 55 |
| | **APPENDICES** | 59 |
| | Appendix A — Annotation Guidelines | 59 |
| | Appendix B — Hyperparameter Configuration | 60 |
| | Appendix C — Sample Source Code | 61 |

---

# LIST OF FIGURES

| Figure | Caption | Page |
|---|---|---:|
| Figure 1.1 | The three behavioural pillars of cyberbullying: aggression, repetition and intent to harm | 4 |
| Figure 3.1 | Overall architecture of the proposed two-stage cyberbullying detection system | 23 |
| Figure 3.2 | Construction of the message-level and pair-level data views | 26 |
| Figure 3.3 | Text preprocessing and Roman Urdu normalisation pipeline | 28 |
| Figure 3.4 | Stage 1 — m-BERT / MuRIL aggression classifier architecture | 33 |
| Figure 3.5 | Repetition and intent scoring over a conversation timeline | 36 |
| Figure 3.6 | Stage 2 — user-pair cyberbullying classifier (dense neural network) | 37 |
| Figure 3.7 | Stage 3 — multimodal image-and-text (ResNet50 + transformer) architecture | 40 |
| Figure 4.1 | Class distribution of messages and user-pairs | 42 |
| Figure 4.2 | Inter-annotator agreement (Fleiss' Kappa) | 43 |
| Figure 4.3 | Stage 1 model comparison (SVM, BiLSTM, m-BERT, MuRIL) | 44 |
| Figure 4.4 | Confusion matrix — m-BERT aggression classifier | 44 |
| Figure 4.5 | Confusion matrix — MuRIL aggression classifier | 45 |
| Figure 4.6 | Confusion matrix — Stage 2 cyberbullying classifier | 46 |
| Figure 4.7 | Stage 2 cross-validation performance across folds | 46 |
| Figure 4.8 | Earlier collapsed model vs corrected model | 47 |

---

# LIST OF TABLES

| Table | Caption | Page |
|---|---|---:|
| Table 2.1 | Summary of related work on cyberbullying and aggression detection | 20 |
| Table 3.1 | Composition of the message-level dataset by language | 25 |
| Table 3.2 | Sample annotated messages from the dataset | 26 |
| Table 3.3 | Pair-level relationship features used by the Stage 2 classifier | 32 |
| Table 3.4 | Evaluation metrics and their definitions | 38 |
| Table 3.5 | Training configuration and hyperparameters | 39 |
| Table 4.1 | Dataset statistics and class distribution | 42 |
| Table 4.2 | Inter-annotator agreement (Fleiss' Kappa) | 43 |
| Table 4.3 | Stage 1 aggression detection results (SVM, BiLSTM, m-BERT, MuRIL) | 44 |
| Table 4.4 | Stage 2 cyberbullying classification results | 45 |
| Table 4.5 | Earlier collapsed model vs corrected two-stage model | 47 |
| Table 4.6 | Comparison of the proposed system with existing work | 50 |

---

# LIST OF ABBREVIATIONS

| Abbreviation | Meaning |
|---|---|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| AUC | Area Under the Curve |
| BERT | Bidirectional Encoder Representations from Transformers |
| BoW | Bag of Words |
| CB | Cyberbullying |
| CNN | Convolutional Neural Network |
| F1 | F1-Score (harmonic mean of precision and recall) |
| FN | False Negative |
| FP | False Positive |
| ICT | Information and Communication Technology |
| LSTM | Long Short-Term Memory |
| m-BERT | Multilingual BERT |
| ML | Machine Learning |
| MLP | Multi-Layer Perceptron |
| MuRIL | Multilingual Representations for Indian Languages |
| NLP | Natural Language Processing |
| RNN | Recurrent Neural Network |
| SGD | Stochastic Gradient Descent |
| SVM | Support Vector Machine |
| TN | True Negative |
| TP | True Positive |
| XAI | Explainable Artificial Intelligence |
