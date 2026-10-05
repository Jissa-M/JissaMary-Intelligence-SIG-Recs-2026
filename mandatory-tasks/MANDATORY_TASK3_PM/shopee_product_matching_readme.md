# Shopee Product Matching

This project develops a product matching system for the Shopee Product Matching dataset.

The objective is to determine whether two product listings represent the **same underlying product** using information from:

- Product titles
- Product images
- Text representations
- Image embeddings
- Similarity measures
- Multimodal fusion

The project is divided into four stages:

1. **Part A — Dataset Exploration**
2. **Part B — Text-Based Product Matching**
3. **Part C — Image-Based Product Matching**
4. **Finale — Multimodal Product Matching**

The progression is intentional:

```text
Dataset Understanding
        ↓
Text-Based Matching
        ↓
Image-Based Matching
        ↓
Multimodal Matching
```

---

## 1. Problem Statement

Online marketplaces contain multiple listings for the same product. Two listings may describe the same product using:

- Different titles
- Different wording
- Different languages
- Different sellers
- Different images
- Different image backgrounds
- Different product presentation

Therefore, simple exact matching is not sufficient. The goal of this project is to build a system that takes two product listings and predicts:

```text
Same Product  OR  Different Product
```

The project investigates whether textual information, visual information, and their combination can be used to make reliable product-matching decisions.

---

## 2. Dataset

The project uses the Shopee Product Matching dataset. The training dataset contains **34,250 product listings**.

### Main Columns

| Column | Description |
| :--- | :--- |
| `posting_id` | Unique identifier for a product listing |
| `image` | Filename of the product image |
| `image_phash` | Perceptual hash of the image |
| `title` | Product title/description |
| `label_group` | Ground-truth product identity |

> **Note:** The `label_group` column is particularly important. Listings with the same `label_group` represent the same underlying product.

---

## 3. Project Structure

```text
Shopee-Product-Matching/
│
├── PartA/
│   ├── notebook.ipynb
│   └── README.md
│
├── PartB/
│   ├── notebook.ipynb
│   ├── README.md
│   └── results/
│
├── PartC/
│   ├── notebook.ipynb
│   ├── README.md
│   └── results/
│
└── Finale/
    ├── notebook.ipynb
    ├── README.md
    ├── src/
    ├── results/
    └── report.pdf
```

---

## 4. Technologies Used

The project was implemented primarily in Python.

### Main Libraries
- **Python**
- **Pandas** & **NumPy**
- **Scikit-learn**
- **PyTorch** & **Torchvision**
- **Matplotlib**

### Machine Learning Techniques
- Exploratory Data Analysis (EDA)
- TF-IDF & Cosine Similarity
- Threshold Optimization
- Pair-based Classification
- CNN Image Embeddings (ResNet18, ResNet50)
- Logistic Regression
- Weighted Score Fusion & Multimodal Learning
- Error Analysis & Ablation Analysis

---

# PART A — DATASET EXPLORATION

## 5. Objective

Part A focuses on understanding the dataset before applying machine learning. The main objectives were:

- Inspect the dataset structure.
- Identify the number of product groups.
- Analyze product-group sizes.
- Study product title lengths.
- Investigate duplicate images and perceptual hashes.
- Examine title variation.
- Understand challenges that may affect product matching.

The goal was to use these observations to guide the design of Parts B, C, and the Finale.

---

## 6. Dataset Statistics

- **Total Listings:** 34,250
- **Unique Product Groups:** 11,014
- **Missing Values:** None in main dataset columns.

---

## 7. Product Group Analysis

The `label_group` column identifies the underlying product.

- **Unique Product Groups:** 11,014
- **Average Group Size:** 3.11 listings
- **Median Group Size:** 2 listings
- **Largest Group Size:** 51 listings

### Group Size Distribution

| Group Size | Number of Groups |
| :--- | :--- |
| **2 listings** | 6,979 |
| **3 listings** | 1,779 |
| **4 listings** | 862 |
| **5+ listings** | 1,394 |

---

## 8. Product Title Analysis

Product titles were analyzed by word count and character count.

### Word Count
- **Mean:** 9.41
- **Median:** 9
- **Minimum:** 1
- **Maximum:** 61

### Character Count
- **Mean:** 56.16
- **Median:** 53
- **Minimum:** 5
- **Maximum:** 357

---

## 9. Image Analysis

- **Unique Image Files:** 32,412
- **Total Listings:** 34,250
- **Duplicate Image Entries:** 1,838

---

## 10. Perceptual Hash Analysis

- **Unique pHashes:** 28,735
- **Duplicate pHash rows:** 5,515
- **Most frequent pHash count:** 26 times

---

## 11. Important Observations from Part A

1. **Multiple listings represent the same product:** Fundamentally an identity-matching problem.
2. **Product groups are usually small:** Median group size is 2, making pairwise comparison a useful formulation.
3. **Titles vary substantially:** Exact string matching is insufficient.
4. **Images contain useful information:** Repeated images and perceptual hashes show strong visual similarity signals.
5. **No single modality is guaranteed to work:** Requires a multimodal approach.

---

# PART B — TEXT-BASED PRODUCT MATCHING

## 12. Objective

Part B investigates whether product titles alone can determine matching products using TF-IDF representations and Cosine Similarity.

```text
Product Titles ──→ TF-IDF Representation ──→ Cosine Similarity ──→ Threshold ──→ Decision
```

---

## 13. Pair Construction

Pairwise dataset setup:
- **`label = 1`**: Same product (from same `label_group`)
- **`label = 0`**: Different products (from different `label_group`)

### Pair Counts
- **Positive pairs:** 83,751
- **Negative pairs:** 83,751
- **Total pairs:** 167,502
- **Split:** 80% train / 20% test (Stratified split)

---

## 14. Word-Level TF-IDF

### Configuration
```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=3,
    max_features=50000
)
```

---

## 15. Word TF-IDF Results

- **Best Threshold:** `0.10`

| Metric | Score |
| :--- | :--- |
| **Precision** | 0.9959 |
| **Recall** | 0.9319 |
| **F1-Score** | 0.9628 |

---

## 16. Character-Level TF-IDF

### Configuration
```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_features=50000
)
```

---

## 17. Character TF-IDF Results

- **Best Threshold:** `0.10`

| Method | Threshold | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Word TF-IDF** | 0.10 | 0.9959 | 0.9319 | 0.9628 |
| **Character TF-IDF** | 0.10 | 0.9909 | 0.9616 | **0.9760** |

---

## 18. Text Error Analysis

At threshold `0.10`, Character TF-IDF produced:
- **Correct:** 32,709
- **False Positives:** 148
- **False Negatives:** 644

```text
Confusion Matrix:
[[16603   148]
 [  644 16106]]
```

---

## 19. Part B Conclusion

Character TF-IDF outpaced Word TF-IDF (**F1 = 0.9760**), but text-only models fail when titles use completely different terminology, multiple languages, or noisy seller descriptions.

---

# PART C — IMAGE-BASED PRODUCT MATCHING

## 20. Objective

Part C evaluates CNN image embeddings (ResNet50 and ResNet18) and cosine similarity to match product images.

```text
Product Image ──→ Pretrained CNN ──→ Image Embedding ──→ Cosine Similarity ──→ Threshold ──→ Decision
```

---

## 21. Computational Constraint & Sampling

- **Environment:** CPU-only (`CUDA available: False`)
- **Sample Size:** 5,000 images (`np.random.seed(42)`)

---

## 22. ResNet50 & ResNet18 Pair Setup

### ResNet50 Evaluation Pairs
- **Positive pairs:** 1,605
- **Negative pairs:** 1,605
- **Total pairs:** 3,210

---

## 23. Image Matching Results

| Model | Embedding Dim | Threshold | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | 2048-D | 0.40 | **91.50%** | 90.76% | **92.40%** | **91.57%** |
| **ResNet18** | 512-D | 0.70 | 89.78% | **91.60%** | 87.60% | 89.55% |

---

## 24. Part C Conclusion

ResNet50 performed best among visual models (**F1 = 0.9157**). While image-only matching was weaker than text matching, visual embeddings capture details absent in text, providing strong support for multimodal fusion.

---

# FINALE — MULTIMODAL PRODUCT MATCHING

## 25. Objective

The Finale combines Character TF-IDF text features and ResNet50 visual embeddings into a score-fusion pipeline.

```text
Product Titles ──→ Character TF-IDF ──→ Text Similarity ──┐
                                                            │
Product Images ──→ ResNet50 ──────────→ Image Similarity ─┤
                                                            ↓
                                                       Score Fusion
                                                            ↓
                                                            Threshold ──→ Decision
```

---

## 26. Final Evaluation Setup

- **Sample Size:** 5,000 images
- **Positive Pairs:** 1,603
- **Negative Pairs:** 1,603
- **Total Pairs:** 3,206

---

## 27. Weighted Score Fusion Experiments

### Formula
$$\text{Final Score} = \alpha \cdot \text{Text Similarity} + (1 - \alpha) \cdot \text{Image Similarity}$$

### Experiment Results

| Text Weight ($\alpha$) | Image Weight ($1-\alpha$) | Threshold | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 1.00 | 0.40 | 0.9205 | 0.9171 | 0.9245 | 0.9208 |
| 0.25 | 0.75 | 0.35 | 0.9585 | 0.9543 | 0.9632 | 0.9587 |
| 0.50 | 0.50 | 0.30 | 0.9797 | 0.9867 | 0.9726 | 0.9796 |
| **0.75** | **0.25** | **0.20** | **0.9863** | **0.9949** | 0.9775 | **0.9862** |
| 1.00 | 0.00 | 0.10 | 0.9779 | 0.9917 | 0.9638 | 0.9775 |

---

## 28. Learned Fusion (Logistic Regression)

Logistic Regression was trained on similarity pairs:
- **Text Similarity Coefficient:** `11.0626`
- **Image Similarity Coefficient:** `7.7032`
- **Intercept:** `-5.1481`
- **Best Probability Threshold:** `0.30`

---

## 29. Final Model Comparison & Ablation

| Model / Fusion Method | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Image-only (ResNet50)** | 0.9205 | 0.9171 | 0.9245 | 0.9208 |
| **Text-only (Char TF-IDF)** | 0.9779 | 0.9917 | 0.9638 | 0.9775 |
| **Multimodal (75% Text / 25% Image)** | **0.9863** | **0.9949** | 0.9775 | **0.9862** |
| **Learned Fusion (Logistic Reg.)** | 0.9860 | 0.9785 | **0.9938** | 0.9861 |

---

## 30. Multimodal Error Analysis

Comparing Multimodal (75/25) vs. Text-Only:
- **Text wrong $\rightarrow$ Multimodal correct:** 32 cases rescued by image signal.
- **Text correct $\rightarrow$ Multimodal wrong:** 5 cases.
- **Net Improvement:** **+27 cases**.

---

## 31. Future Improvements

1. **Semantic Text Embeddings:** Replace TF-IDF with Sentence-BERT or Multilingual Transformers.
2. **Fine-Tuned Image Embeddings:** Fine-tune CNNs using Triplet / Contrastive Loss specifically on product pairs.
3. **Advanced Fusion:** Cross-modal attention or Transformer-based multimodal fusion.
4. **Candidate Retrieval & Reranking:** Multi-stage retrieval (Top-K) to scale to millions of product pairs.
5. **Group-Level Data Splitting:** Evaluate on completely unseen `label_group` instances to test true out-of-sample generalization.

---

## 32. Complete Project Pipeline

```text
SHOPEE PRODUCT MATCHING
          │
          ▼
PART A — EDA (34,250 Listings / 11,014 Groups)
          │
  ┌───────┴────────┐
  ▼                ▼
Titles          Images & pHash
  │                │
  ▼                ▼
PART B — TEXT   PART C — IMAGE
(Char TF-IDF)   (ResNet50)
F1 = 0.9760     F1 = 0.9157
  │                │
  └───────┬────────┘
          ▼
FINALE — MULTIMODAL FUSION
(75% Text + 25% Image)
          │
          ▼
     F1 = 0.9862
```