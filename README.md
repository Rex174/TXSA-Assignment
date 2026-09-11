# CT107-3-3 TXSA — Text Analytics and Sentiment Analysis
### Group Assignment (Group 21) — Asia Pacific University (APU)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5.svg)](https://spacy.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

> **Module:** CT107-3-3 Text Analytics and Sentiment Analysis (TXSA)
> **Programme:** BSc (Hons) Computer Science — Data Analytics
> **Institution:** Asia Pacific University of Technology & Innovation, Bukit Jalil, Malaysia

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [My Contributions](#my-contributions)
- [Part A — Text Pre-processing and Basic NLP](#part-a--text-pre-processing-and-basic-nlp)
  - [Q2 — Word Stemming](#q2--word-stemming-my-contribution)
  - [Q5 — Alternative Tokenization using spaCy](#q5--alternative-tokenization-using-spacy-my-contribution)
- [Part B — Supervised Text Classification](#part-b--supervised-text-classification)
  - [Literature Review](#literature-review-my-contribution)
  - [Logistic Regression Model](#logistic-regression-model-my-contribution)
- [Results Summary](#results-summary)
- [Key Achievements](#key-achievements)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Team & Workload Matrix](#team--workload-matrix)
- [References](#references)

---

## Overview

This repository contains the complete implementation for the **CT107-3-3 TXSA Group Assignment**, a two-part project covering the full spectrum of text analytics — from foundational linguistic pre-processing to the development and optimisation of supervised machine learning models for sentiment classification.

The assignment is split into two major components:

| Part | Focus | Marks | Dataset |
|---|---|---|---|
| **Part A** | Text pre-processing, tokenization, stemming, POS tagging, parse trees, and bigram language models | 50 | `Data_1.txt`, `Data_2.txt`, `Data_3.txt` |
| **Part B** | Supervised text classification, EDA, model building, hyperparameter tuning, and comparative evaluation | 50 | IMDB Dataset of 50K Movie Reviews |

**Part B Objective:** Build an effective Supervised Text Classification model for binary sentiment prediction — classifying IMDB movie reviews as either **Positive** or **Negative**. Each of the four group members independently implemented a different classification model, and all four were evaluated and compared to identify the best performer.

---

## Repository Structure

```
TXSA-Assignment/
│
├── Part_A_G20.ipynb                      # Compiled Part A code (all members)
├── Part_B_G20.ipynb                      # Compiled Part B code (all members)
├── TXSA_G20_Part_A__Part_B_Report.pdf    # Full written report (117 pages)
├── CT107-3-3-TXSA_-_Group_Assignment.docx # Assignment question document
│
├── data/
│   ├── Data_1.txt                        # Corpus for Part A Q1 & Q2
│   ├── Data_2.txt                        # Corpus for Part A Q3
│   ├── Data_3.txt                        # Corpus for Part A Q4
│   └── IMDB Dataset (cleaned).csv        # Cleaned IMDB dataset for Part B
│
└── README.md
```

---

## My Contributions

**Ramaneiss Pillai A/L S.Gopalan — TP070818**

My individual responsibilities across both parts of the assignment were as follows:

| Part | Section | Task |
|---|---|---|
| **Part A** | `2.2 (Q2)` | **Word Stemming** — Regular Expression, Porter, and Lancaster stemmers |
| **Part A** | `2.5.2 (Q5)` | **Alternative Tokenization using spaCy** (individual component) |
| **Part B** | `3.1.3` | **Literature Review** — comprehensive 4-domain academic review |
| **Part B** | `3.6` | **Logistic Regression Model** — baseline build, evaluation, and hyperparameter tuning |

---

## Part A — Text Pre-processing and Basic NLP

### Q2 — Word Stemming *(My Contribution)*

Stemming reduces inflected words to their root form (e.g., *classification*, *classified*, *classifying* → *classif*), which is a critical pre-processing step for reducing vocabulary size and normalising features before model training.

#### What Was Implemented

Three distinct stemming algorithms were applied to the tokenised corpus from `Data_1.txt` and compared side by side:

| Stemmer | Approach | Aggressiveness |
|---|---|---|
| **Regular Expression Stemmer** | Rule-based pattern matching using user-defined suffix regex | Fully customisable, but naive |
| **Porter Stemmer** | Five-phase sequential suffix-stripping algorithm | Moderate — conservative and readable output |
| **Lancaster Stemmer** | Iterative rule application until no further reduction | Highly aggressive — shortest stems |

#### Key Findings

- **Regular Expression Stemmer** offered maximum control but failed on words whose suffixes were not explicitly covered by the defined pattern, producing inconsistent results across the corpus.
- **Porter Stemmer** provided the best balance — it produced linguistically sensible stems that remained human-readable, making it the most practical choice for general text analytics tasks.
- **Lancaster Stemmer** over-stemmed aggressively, frequently reducing words to stems so short they lost semantic meaning entirely, which risks conflating unrelated words into a single feature.

#### Importance of Stemming — Documented Justification

1. **Vocabulary and Dimensionality Reduction** — consolidates word variants into a single feature, reducing the feature space and mitigating overfitting.
2. **Improved Information Retrieval** — a query for *"classify"* can successfully retrieve documents containing *"classification"* or *"classified"*, boosting recall.
3. **Better Feature Normalisation for ML** — provides consistent input for algorithms such as Naive Bayes, SVM, and TF-IDF vectorisation by preventing information fragmentation across word variants.

---

### Q5 — Alternative Tokenization using spaCy *(My Contribution)*

The **Individual Component** required each student to independently implement an alternative tokenization technique and critically compare it against the group's chosen approach (NLTK `word_tokenize`).

#### My Chosen Alternative: **spaCy** (`en_core_web_sm`)

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(paragraph)
spacy_tokens = [token.text for token in doc]
```

**Result:** spaCy produced **97 tokens** from `Data_1.txt`.

#### Comparison: spaCy vs NLTK

| Feature | NLTK `word_tokenize` | spaCy `en_core_web_sm` |
|---|---|---|
| Token count (`Data_1.txt`) | Fewer tokens | **97 tokens** |
| Newline / whitespace | Discarded silently | Retained as `'\n'` token |
| `"open-class"` handling | Kept as **one** token | Split into `['open', '-', 'class']` |
| Standard punctuation (`.`, `,`, `;`) | Separated | Separated |
| Underlying mechanism | Rule-based Punkt tokenizer | Linguistic model + full NLP pipeline |
| External model required | No | Yes (~12 MB download) |

#### Verdict: **Different — Not Strictly Better or Worse**

**Arguments in favour of spaCy:**
- **Integrated NLP pipeline** — a single `nlp()` call simultaneously performs tokenisation, POS tagging, dependency parsing, and Named Entity Recognition.
- **Linguistically principled token boundaries** — explicit prefix, suffix, and infix rules constructed by computational linguists make it robust on URLs, emails, and special characters.
- **Lossless tokenisation** — every character in the source text is recoverable from the token sequence.

**Arguments against spaCy for this specific task:**
- **Whitespace noise** — the retained `'\n'` token carries no semantic value for classification and requires an extra filtering step.
- **Hyphen splitting** — decomposing `"open-class"` loses the compound meaning of an established NLP technical term.
- **Higher resource cost** — requires a ~12 MB trained model versus NLTK's few-kilobyte `punkt` data file, which is a real constraint in offline or resource-limited environments.

**Conclusion:** spaCy is the clearly superior choice when multiple NLP sub-tasks are required together, but for the narrow scope of tokenisation, stopword removal, and punctuation filtering alone, NLTK is equally effective and considerably more lightweight.

---

## Part B — Supervised Text Classification

### Dataset

**IMDB Dataset of 50K Movie Reviews** — [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/)

| Property | Value |
|---|---|
| Original size | 50,000 reviews |
| Duplicates removed | 418 |
| **Final cleaned size** | **49,582 reviews** |
| Class distribution | 24,884 Positive / 24,698 Negative (balanced) |
| Missing values | None |
| Columns | `review` (text), `sentiment` (string), `label` (0 = Negative, 1 = Positive) |

---

### Literature Review *(My Contribution)*

I authored the complete **comprehensive literature review** for Part B Q1, structuring it into **four distinct research domains** spanning two decades of sentiment analysis research. The review was deliberately written to cover the full methodological landscape so that it contextualises **all four models** implemented by the group, rather than favouring any single approach.

#### Domain 1 — Early Foundations: Lexicon-Based and Classical Machine Learning

Traced the field's origins to **Pang, Lee & Vaithyanathan (2002)**, whose seminal EMNLP paper *"Thumbs Up? Sentiment Classification using Machine Learning Techniques"* first demonstrated that Naïve Bayes, Maximum Entropy, and SVM could outperform human-produced baselines on movie review polarity. Covered **Maas et al. (2011)**, who introduced the IMDB Large Movie Review Dataset used in this very assignment and pioneered sentiment-optimised word vectors.

#### Domain 2 — Word Embeddings and Distributed Text Representations

Documented the pivotal shift from sparse to dense representations via **Word2Vec (Mikolov et al., 2013)** with its CBOW and Skip-gram architectures, and **GloVe (Pennington, Socher & Manning, 2014)**, which leverages global co-occurrence statistics. Explained why placing semantically similar sentiment words close together in embedding space allows classifiers to generalise across synonymous expressions.

#### Domain 3 — Deep Learning Approaches and Advances

Covered **Kim (2014)** on Convolutional Neural Networks for sentence classification, the evolution of RNN and **LSTM** architectures with their gated memory cells for sequential context, and the subsequent emergence of transformer-based models. Critically noted that RNN-family models compute sequentially and cannot be parallelised — a limitation directly observed in this project's LSTM implementation.

#### Domain 4 — Hyperparameter Tuning and Model Optimisation

Reviewed the literature on systematic model optimisation, covering GridSearchCV versus RandomizedSearchCV trade-offs, cross-validation strategies, and the role of regularisation strength in high-dimensional sparse feature spaces — directly informing the tuning methodology applied across all four models in Part B.

---

### Logistic Regression Model *(My Contribution)*

My chosen supervised classification model for Part B. **This model achieved the best performance of all four models tested by the group.**

#### Architecture Pipeline

```
Raw Review Text (49,582 reviews)
        ↓
80/20 Stratified Train-Test Split
   (39,665 train / 9,917 test)
        ↓
TF-IDF Vectorization
   (unigrams + bigrams, 50,000 features)
        ↓
Logistic Regression Classifier
   (lbfgs solver, L2 regularisation, balanced class weights)
        ↓
Binary Sentiment Prediction (Positive / Negative)
```

#### Baseline Configuration

| Component | Hyperparameter | Value |
|---|---|---|
| TF-IDF | `max_features` | 50,000 |
| TF-IDF | `ngram_range` | (1, 2) — unigrams + bigrams |
| TF-IDF | `sublinear_tf` | True |
| TF-IDF | `min_df` / `max_df` | 2 / 0.95 |
| Logistic Regression | `C` | 1.0 |
| Logistic Regression | `solver` | lbfgs |
| Logistic Regression | `class_weight` | balanced |

#### Baseline Results

| Metric | Score |
|---|---|
| Accuracy | **0.9061** (90.61%) |
| Precision | 0.8953 |
| Recall | 0.9206 |
| F1-Score | 0.9078 |
| ROC-AUC | 0.9680 |
| Training time | 5.85s |

**Confusion Matrix (Baseline):** TN = 4,404 · FP = 536 · FN = 395 · TP = 4,582

#### Model Interpretability

A key advantage of Logistic Regression is its transparency. By extracting the learned coefficients, the model's decision logic can be directly inspected:

| Strongest Positive Indicators | Coeff | Strongest Negative Indicators | Coeff |
|---|---|---|---|
| great | +8.3888 | bad | −8.8586 |
| excellent | +6.4646 | worst | −8.3971 |
| perfect | +5.2569 | awful | −7.4133 |
| wonderful | +4.8300 | the worst | −6.5772 |
| amazing | +4.6968 | boring | −6.5367 |

The presence of bigrams such as `"the best"` (+4.2021) and `"the worst"` (−6.5772) validates the decision to include bigrams in the n-gram range.

#### Hyperparameter Tuning — RandomizedSearchCV

| Setting | Value |
|---|---|
| Method | `RandomizedSearchCV` |
| Search space | **2,592** possible combinations |
| Iterations sampled | 20 |
| Cross-validation | 5-fold Stratified CV |
| Scoring metric | F1-Score |
| Total model fits | 100 |
| Search runtime | 4,892s (**81.5 minutes**) |

**Hyperparameters Tuned:**

| Hyperparameter | Values Tested |
|---|---|
| `tfidf__max_features` | 30,000 / 50,000 / 70,000 |
| `tfidf__ngram_range` | (1,1) / (1,2) / (1,3) |
| `tfidf__sublinear_tf` | True / False |
| `tfidf__min_df` | 1 / 2 / 3 / 5 |
| `lr__C` | 0.01 / 0.1 / 0.5 / 1.0 / 5.0 / 10.0 |
| `lr__solver` | lbfgs / liblinear / saga |
| `lr__class_weight` | None / balanced |

**Best Configuration Found** (CV F1 = 0.9113):

| Hyperparameter | Baseline | Tuned | Changed |
|---|---|---|---|
| `max_features` | 50,000 | **70,000** | Yes |
| `ngram_range` | (1, 2) | (1, 2) | No |
| `sublinear_tf` | True | **False** | Yes |
| `min_df` | 2 | 2 | No |
| `C` | 1.0 | **10.0** | Yes |
| `solver` | lbfgs | lbfgs | No |
| `class_weight` | balanced | balanced | No |

#### Baseline vs Tuned Performance

| Metric | Baseline | Tuned | Improvement |
|---|---|---|---|
| Accuracy | 0.9061 | **0.9123** | **+0.0062** |
| Precision | 0.8953 | **0.9067** | **+0.0114** |
| Recall | 0.9206 | 0.9198 | −0.0008 |
| F1-Score | 0.9078 | **0.9132** | **+0.0054** |
| ROC-AUC | 0.9680 | **0.9714** | **+0.0034** |

**Confusion Matrix (Tuned):** TN = 4,469 · FP = **471** · FN = 399 · TP = 4,578

Tuning reduced False Positives from 536 to 471 — a **65-error reduction** — which directly drove the Precision gain of +0.0114. The negligible Recall decrease (−0.0008) represents an acceptable trade-off for the improvements achieved across all other metrics.

---

## Results Summary

### Final Model Comparison — All Four Group Models

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|:---:|---|---|---|---|---|
| **1** | **Logistic Regression (Tuned)** | **91.23%** | **0.9067** | **0.9198** | **0.9132** |
| 2 | SVM (Tuned) | 90.17% | 0.8950 | 0.9110 | 0.9029 |
| 3 | Random Forest (Baseline) | 87.08% | 0.8569 | 0.8915 | 0.8739 |
| 4 | LSTM (Baseline) | 85.25% | 0.8288 | 0.8899 | 0.8583 |

### Why Logistic Regression Won

- **Optimal fit for sparse high-dimensional data** — TF-IDF produces a feature space where linear decision boundaries separate classes effectively, which suits Logistic Regression's mathematical formulation perfectly.
- **Random Forest limitation** — tree-based ensembles struggle with sparse TF-IDF matrices where the vast majority of feature values are near-zero across documents.
- **LSTM limitation** — underperformed at baseline due to the absence of pre-trained word embeddings, restricted training epochs, and the prohibitive computational cost of tuning on long review sequences.
- **Broader finding** — this result aligns with established literature showing that classical linear models frequently match or outperform deep learning on binary sentiment tasks when using bag-of-words and TF-IDF representations, unless very large datasets and high-quality pre-trained embeddings are available.

---

## Key Achievements

- **Best-performing model in the group** — Logistic Regression achieved the highest scores across every single evaluation metric (Accuracy, Precision, Recall, F1, ROC-AUC).
- **Crossed the 91% accuracy threshold** on a 49,582-review dataset with fully interpretable, inspectable model coefficients.
- **Systematic hyperparameter optimisation** — searched a 2,592-combination space via 100 cross-validated model fits over 81.5 minutes, yielding consistent improvements across four of five metrics.
- **Highly efficient** — 5.85-second baseline training time versus hours required for the deep learning alternatives, demonstrating a strong performance-to-cost ratio.
- **Authored the full 4-domain academic literature review**, contextualising two decades of sentiment analysis research across all methodological families relevant to the group's four models.
- **Delivered a critical comparative analysis** of spaCy vs NLTK tokenisation, reaching a nuanced, evidence-backed conclusion rather than a simplistic better/worse verdict.

---

## Tech Stack

**Language:** Python 3.10+

| Category | Libraries |
|---|---|
| **NLP** | `nltk`, `spacy` (`en_core_web_sm`), `re`, `textblob` |
| **Machine Learning** | `scikit-learn` (TfidfVectorizer, LogisticRegression, RandomizedSearchCV, StratifiedKFold, Pipeline) |
| **Data Handling** | `pandas`, `numpy`, `scipy` |
| **Visualisation** | `matplotlib`, `seaborn` |
| **Environment** | Jupyter Notebook, Anaconda |

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Rex174/TXSA-Assignment.git
cd TXSA-Assignment
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy textblob jupyter
```

### 3. Download required language models and corpora

```bash
# spaCy English model (required for Part A Q5)
python -m spacy download en_core_web_sm

# NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

### 4. Prepare the dataset

Download the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle and place the CSV in the `data/` directory.

### 5. Launch Jupyter and run the notebooks

```bash
jupyter notebook
```

Open `Part_A_G20.ipynb` or `Part_B_G20.ipynb` and run all cells sequentially.

> **Note on runtime:** The `RandomizedSearchCV` cell in the Logistic Regression section takes approximately **80–85 minutes** to complete on a standard CPU. Reduce `N_ITER` from 20 to a lower value to shorten the search if needed.

---

## Team & Workload Matrix

| Name | Student ID | Part A | Part B |
|---|---|---|---|
| Muhammad Irfan bin Mohd Rizal | TP078491 | Q1, Q5 (scikit-learn) | LSTM Model, Conclusion & Comparison |
| Sarvein Rao A/L Sathiah | TP071496 | Q3, Q5 (TextBlob) | SVM Model, EDA & Pre-Processing |
| Voon Kai Wen | TP077157 | Q4, Q5 (Hugging Face) | Random Forest Model, EDA |
| **Ramaneiss Pillai A/L S.Gopalan** | **TP070818** | **Q2 (Stemming), Q5 (spaCy)** | **Literature Review, Logistic Regression Model** |

---

## References

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL 2011, pp. 142–150. http://ai.stanford.edu/~amaas/data/sentiment/
- Pang, B., Lee, L., & Vaithyanathan, S. (2002). *Thumbs up? Sentiment Classification using Machine Learning Techniques.* EMNLP 2002, pp. 79–86. https://aclanthology.org/W02-1011/
- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* EMNLP 2014, pp. 1746–1751. https://aclanthology.org/D14-1181/
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* https://arxiv.org/abs/1301.3781
- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP 2014, pp. 1532–1543. https://nlp.stanford.edu/pubs/glove.pdf
- IMDB Dataset of 50K Movie Reviews (2019). Kaggle. https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## Academic Integrity

This repository is published for **portfolio and educational reference purposes only**. It represents coursework submitted for assessment at Asia Pacific University. Any reuse must comply with your own institution's academic integrity policies — do not submit this work, in whole or in part, as your own.

---

<div align="center">

**Ramaneiss Pillai A/L S.Gopalan — TP070818**
BSc (Hons) Computer Science (Data Analytics) · Asia Pacific University

</div>
