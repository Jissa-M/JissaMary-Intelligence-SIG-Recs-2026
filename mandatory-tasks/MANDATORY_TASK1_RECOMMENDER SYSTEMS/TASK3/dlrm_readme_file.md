# Deep Learning Recommendation Model (DLRM) Implementation Report

This repository contains the complete report, architecture breakdown, evaluation metrics, and ablation study for the custom **Deep Learning Recommendation Model (DLRM)** implementation on click-through rate (CTR) prediction data.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Objective](#2-objective)
3. [Dataset](#3-dataset)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Numerical Feature Processing](#5-numerical-feature-processing)
6. [Categorical Feature Processing](#6-categorical-feature-processing)
7. [DLRM Architecture](#7-dlrm-architecture)
8. [Input Layers](#8-input-layers)
9. [Bottom MLP](#9-bottom-mlp)
10. [Categorical Embeddings](#10-categorical-embeddings)
11. [Why Embeddings Are Used](#11-why-embeddings-are-used)
12. [Feature Interaction](#12-feature-interaction)
13. [Number of Pairwise Interactions](#13-number-of-pairwise-interactions)
14. [Why Explicit Interactions Matter](#14-why-explicit-interactions-matter)
15. [Interaction Vector](#15-interaction-vector)
16. [Top MLP](#16-top-mlp)
17. [Complete DLRM Architecture Flowchart](#17-complete-dlrm-architecture-flowchart)
18. [Model Construction](#18-model-construction)
19. [Compilation](#19-compilation)
20. [Training](#20-training)
21. [Early Stopping](#21-early-stopping)
22. [Test Preprocessing](#22-test-preprocessing)
23. [Evaluation Metrics](#23-evaluation-metrics)
24. [Threshold Selection](#24-threshold-selection)
25. [DLRM Results](#25-dlrm-results)
26. [Calibration](#26-calibration)
27. [Ablation Study — Removing Interactions](#27-ablation-study--removing-interactions)
28. [Ablation Results](#28-ablation-results)
29. [DLRM vs. Vanilla Neural Network](#29-dlrm-vs-vanilla-neural-network)
30. [Why DLRM Can Still Be Architecturally Superior](#30-why-dlrm-can-still-be-architecturally-superior)
31. [DLRM vs. Matrix Factorization](#31-dlrm-vs-matrix-factorization)
32. [Advantages of DLRM](#32-advantages-of-dlrm)
33. [Limitations and Costs](#33-limitations-and-costs)
34. [Key Implementation Decisions Summary](#34-key-implementation-decisions-summary)
35. [Conclusion](#35-conclusion)

---

## 1. Overview
The Deep Learning Recommendation Model (DLRM) implementation utilizes:
* **13 numerical features**
* **26 categorical features**
* **Embedding tables** for categorical features
* **A Bottom MLP** for numerical features
* **Explicit pairwise feature interactions**
* **A Top MLP** for final prediction
* **Binary cross-entropy loss**
* **Evaluation metrics:** ROC-AUC, PR-AUC, log loss, and threshold-based metrics
* **Calibration analysis**
* **An ablation study** removing explicit interactions

The same CTR dataset and preprocessing/split policy used in Task 2 are reused here so that the two models can be compared fairly.

---

## 2. Objective
The objective of this task is to implement the core architecture of DLRM without using a pretrained DLRM model or a ready-made DLRM implementation.

The implementation should:
* Parse the 13 numerical and 26 categorical fields.
* Prevent validation/test information from leaking into preprocessing.
* Build embedding tables for categorical features.
* Process numerical features using an MLP.
* Explicitly calculate pairwise interactions between feature representations.
* Combine the interactions with the dense representation.
* Use a top MLP for binary click prediction.
* Compare DLRM against the vanilla neural network from Task 2.
* Perform an ablation study by removing explicit interactions.
* Analyze model performance, parameter count, training cost, and calibration.

---

## 3. Dataset
The same CTR dataset used in Task 2 is used for Task 3.

The dataset contains:
* **13 numerical features**
* **26 categorical features**
* **1 binary target variable:** `label` (represents whether a click occurred)

```text
Numerical features   : 13
Categorical features : 26
Target               : label
```

The data is loaded using:
```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

The training data is divided into training and validation sets using:
```python
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This produces:
* **Training samples:** 32,000
* **Validation samples:** 8,000
* **Test samples:** 10,000

The same split and random seed policy as Task 2 are maintained to make the comparison meaningful.

---

## 4. Data Preprocessing

### 4.1 Identifying Feature Types
The numerical features are identified using the `integer_` prefix:
```python
numeric_cols = [
    col for col in train.columns
    if col.startswith("integer_")
]
```

The categorical features are identified using:
```python
categorical_cols = [
    col for col in train.columns
    if col.startswith("categorical_")
]
```

**Summary:**
* 13 numerical features
* 26 categorical features

---

## 5. Numerical Feature Processing
Numerical features are processed using two steps:
1. Missing-value imputation
2. Standardization

### 5.1 Missing Values
Missing numerical values are replaced using the median of the training data:
```python
numeric_imputer = SimpleImputer(strategy="median")
X_train_numeric = numeric_imputer.fit_transform(X_train[numeric_cols])
X_val_numeric = numeric_imputer.transform(X_val[numeric_cols])
```
* The imputer is fitted only on the training data.
* The validation and test data are transformed using the already-fitted imputer.
* This prevents information from the validation or test set from influencing preprocessing.

### 5.2 Standardization
The numerical features are standardized using `StandardScaler`:
```python
numeric_scaler = StandardScaler()
X_train_numeric = numeric_scaler.fit_transform(X_train_numeric)
X_val_numeric = numeric_scaler.transform(X_val_numeric)
```

The same fitted scaler is later applied to the test data.

**Resulting numerical shapes:**
* **Training numerical shape:** `(32000, 13)`
* **Validation numerical shape:** `(8000, 13)`
* **Test numerical shape:** `(10000, 13)`

---

## 6. Categorical Feature Processing
Categorical features cannot be directly passed into the neural network. Each categorical value is converted into an integer index.

Missing categorical values are first replaced with: `"MISSING"`

Values are converted to strings and a vocabulary is constructed using training data only. For every categorical feature:
```python
train_values = X_train[col].fillna("MISSING").astype(str)
categories = train_values.unique()
category_to_index = {
    category: index + 1
    for index, category in enumerate(categories)
}
```

* The index starts at `1`.
* Index `0` is reserved for unknown categories.

**Mapping Rules:**
* Known category $\rightarrow$ integer index
* Unknown category $\rightarrow$ `0`
* Missing category $\rightarrow$ `"MISSING"`

The same mapping learned from the training set is used for validation and test data. This prevents category information from the validation/test sets from leaking into training.

**Final categorical shapes:**
* **Training categorical shape:** `(32000, 26)`
* **Validation categorical shape:** `(8000, 26)`
* **Test categorical shape:** `(10000, 26)`

---

## 7. DLRM Architecture

```text
                 ┌─────────────────────┐
                 │ Numerical Features  │
                 │      13 features    │
                 └──────────┬──────────┘
                            │
                            ▼
                     Bottom MLP
                      64 → 32
                            │
                            ▼
                    Dense Representation
                            │
                            │
                            │
Categorical Features        │
26 fields                   │
     │                      │
     ▼                      │
26 Embedding Tables         │
     │                      │
     ▼                      │
26 Embedding Vectors        │
     │                      │
     └──────────┬───────────┘
                │
                ▼
        Pairwise Interactions
                │
                ▼
       Interaction Vector
                │
                ▼
      Dense + Interactions
                │
                ▼
              Top MLP
             128 → 64
                │
                ▼
          Sigmoid Output
                │
                ▼
        Click Probability
```

---

## 8. Input Layers
Two separate inputs are used:

### Numerical Input
```python
numeric_input = layers.Input(
    shape=(len(numeric_cols),),
    name="numeric_input"
)
```
* Since there are 13 numerical features: **Input shape = (13,)**

### Categorical Input
```python
categorical_input = layers.Input(
    shape=(len(categorical_cols),),
    dtype="int32",
    name="categorical_input"
)
```
* Since there are 26 categorical features: **Input shape = (26,)**

The categorical values are integer indices that are used to look up embedding vectors.

---

## 9. Bottom MLP
The numerical features are processed using a small MLP:
```python
dense = layers.Dense(
    64,
    activation="relu",
    name="bottom_dense_1"
)(numeric_input)

dense = layers.Dense(
    32,
    activation="relu",
    name="bottom_dense_2"
)(dense)
```

**Transformation sequence:**
$$13 \text{ numerical features} \xrightarrow{\text{Dense}(64, \text{ReLU})} \text{Dense}(64) \xrightarrow{\text{Dense}(32, \text{ReLU})} 32\text{-dimensional dense representation}$$

The numerical features are therefore converted from **13 dimensions** to **32 dimensions**. This 32-dimensional representation participates in the interaction stage.

---

## 10. Categorical Embeddings
Each categorical feature receives its own embedding table.

Set embedding dimension:
```python
embedding_dim = 32
```

For each categorical feature:
```python
embedding = layers.Embedding(
    input_dim=vocabulary_size,
    output_dim=embedding_dim,
    name=f"dlrm_embedding_{i}"
)(categorical_input[:, i])
```

Each categorical field produces a vector of **32 dimensions**:
* `categorical_1` $\rightarrow$ 32-dimensional vector
* `categorical_2` $\rightarrow$ 32-dimensional vector
* `categorical_3` $\rightarrow$ 32-dimensional vector
* ...
* `categorical_26` $\rightarrow$ 32-dimensional vector

Together with the numerical representation, there are:
$$\text{1 dense vector} + \text{26 categorical embedding vectors} = \mathbf{27\text{ \textbf{feature vectors}}}$$

---

## 11. Why Embeddings Are Used
Categorical variables may have a large number of unique values. Representing them using one-hot encoding would create extremely large sparse vectors. Embeddings instead represent each category using a dense learned vector:
* Category A $\rightarrow$ `[0.12, -0.43, 0.71, ...]`
* Category B $\rightarrow$ `[0.55,  0.11, -0.29, ...]`
* Category C $\rightarrow$ `[-0.21, 0.67, 0.04, ...]`

During training, the embedding vectors are updated so that categories useful for predicting clicks can develop useful representations.

---

## 12. Feature Interaction
One of the main differences between this model and the vanilla neural network is the explicit interaction operation.

There are 27 feature vectors:
* **1** numerical representation
* **26** categorical embeddings

The model calculates every unique pairwise interaction. For two vectors $A$ and $B$, the interaction is their dot product:
$$\text{interaction}(A, B) = A \cdot B = \sum_{k} A_k B_k$$

This produces a single scalar for each pair.

---

## 13. Number of Pairwise Interactions
There are 27 feature vectors. The number of unique pairs is calculated as:
$$\frac{27 \times 26}{2} = 351$$

Therefore, the implementation creates **351 pairwise interactions**.

**Code Implementation:**
```python
all_features = [dense] + embedded_features
interaction_features = []

for i in range(len(all_features)):
    for j in range(i + 1, len(all_features)):
        interaction = layers.Dot(
            axes=1,
            name=f"interaction_{i}_{j}"
        )([
            all_features[i],
            all_features[j]
        ])
        interaction_features.append(interaction)
```

**Output:** Number of pairwise interactions = 351

---

## 14. Why Explicit Interactions Matter
CTR prediction often depends on combinations of features rather than individual features, such as:
* User $\times$ Advertisement
* User $\times$ Campaign
* Device $\times$ Advertisement
* User $\times$ Device
* Campaign $\times$ Device

A model that explicitly represents interactions can capture these relationships more directly. The DLRM interaction stage allows the model to learn relationships between:
* Numerical representation and categorical embeddings
* Categorical embedding and categorical embedding

---

## 15. Interaction Vector
The 351 scalar interaction values are concatenated:
```python
interaction_vector = layers.Concatenate(
    name="interaction_vector"
)(interaction_features)
```
* **Interaction vector shape:** 351

The model also retains the original dense numerical representation:
* **Dense representation:** 32

These are combined:
```python
dlrm_input = layers.Concatenate(
    name="dlrm_input"
)([
    dense,
    interaction_vector
])
```
$$\text{DLRM Input Dimension} = 32 + 351 = 383 \text{ dimensions}$$

```text
32-dimensional dense representation
              +
351 interaction values
              ↓
        383-dimensional vector
```

---

## 16. Top MLP
The combined representation is passed through the top MLP:

```python
top = layers.Dense(
    128,
    activation="relu",
    name="top_dense_1"
)(dlrm_input)

top = layers.Dropout(
    0.2,
    name="top_dropout"
)(top)

top = layers.Dense(
    64,
    activation="relu",
    name="top_dense_2"
)(top)

output = layers.Dense(
    1,
    activation="sigmoid",
    name="click_probability"
)(top)
```

**Architecture Pipeline:**
$$383 \xrightarrow{\text{Dense}(128, \text{ReLU})} \text{Dense}(128) \xrightarrow{\text{Dropout}(0.2)} \text{Dropout} \xrightarrow{\text{Dense}(64, \text{ReLU})} \text{Dense}(64) \xrightarrow{\text{Dense}(1, \text{Sigmoid})} \text{Sigmoid Output}$$

The final sigmoid output represents the predicted probability of a click ($\text{Output} \in [0, 1]$).

---

## 17. Complete DLRM Architecture Flowchart

```text
                    INPUT
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
  13 Numerical                26 Categorical
     Fields                       Fields
        │                           │
        ▼                           ▼
   Bottom MLP                26 Embedding Tables
    13 → 64                       │
       → 32                       │
        │                         │
        │                   26 × 32-D vectors
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
             27 feature vectors
                     │
                     ▼
            Pairwise Interactions
                     │
                     ▼
              351 interaction
                  values
                     │
                     ├──────────────┐
                     │              │
                     ▼              │
              Interaction Vector    │
                   351              │
                     │              │
                     └──────┬───────┘
                            ▼
                   Dense + Interaction
                       32 + 351
                         = 383
                            │
                            ▼
                       Dense(128)
                            │
                       Dropout(0.2)
                            │
                        Dense(64)
                            │
                        Sigmoid
                            │
                            ▼
                    Click Probability
```

---

## 18. Model Construction
The complete model is created using:
```python
dlrm_model = Model(
    inputs=[
        numeric_input,
        categorical_input
    ],
    outputs=output,
    name="DLRM"
)
```

* **Total parameters:** ~2,900,513
* **Model size:** ~11.06 MB
* All parameters are trainable.
* Most parameters originate from the categorical embedding tables.

---

## 19. Compilation
The model is trained for binary classification.

* **Loss function:** `loss="binary_crossentropy"`
* **Optimizer:**
  ```python
  tf.keras.optimizers.Adam(learning_rate=0.001)
  ```
* **Evaluation metrics:**
  ```python
  tf.keras.metrics.AUC(name="roc_auc")
  tf.keras.metrics.AUC(name="pr_auc", curve="PR")
  tf.keras.metrics.BinaryAccuracy(name="accuracy")
  ```

---

## 20. Training
The model is trained using:
```python
history = dlrm_model.fit(
    [X_train_numeric, X_train_categorical],
    y_train,
    validation_data=(
        [X_val_numeric, X_val_categorical],
        y_val
    ),
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)
```

**Training Configuration Summary:**
* **Optimizer:** Adam
* **Learning rate:** 0.001
* **Loss:** Binary Cross-Entropy
* **Batch size:** 256
* **Maximum epochs:** 20
* **Early stopping:** Enabled
* **Monitor:** Validation PR-AUC
* **Patience:** 3

The best model weights are restored after early stopping.

---

## 21. Early Stopping
Early stopping is based on validation PR-AUC:
```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)
```
PR-AUC is particularly useful for CTR prediction because clicks are relatively rare. Instead of simply monitoring accuracy, the model tracks the quality of its positive-class predictions.

---

## 22. Test Preprocessing
The test set is processed using the transformations learned from training data.

**Numerical Data:**
```python
X_test_numeric = numeric_imputer.transform(X_test[numeric_cols])
X_test_numeric = numeric_scaler.transform(X_test_numeric)
```

**Categorical Data:**
Categorical data uses category mappings derived from the training set. Unknown test categories are mapped to `0`.

**Final Shapes:**
* **Test numerical shape:** `(10000, 13)`
* **Test categorical shape:** `(10000, 26)`
* **Test labels shape:** `(10000,)`

---

## 23. Evaluation Metrics
The model is evaluated using:

* **ROC-AUC:** Measures the ability of the model to rank positive examples above negative examples (Higher is better).
* **PR-AUC:** Measures precision-recall performance, which is essential for imbalanced CTR data (Higher is better).
* **Log Loss:** Measures the quality of predicted probabilities (Lower is better).
* **Accuracy:** Measures the fraction of correctly classified examples after applying a threshold.
* **Precision:** Measures how many predicted clicks were actually clicks.
* **Recall:** Measures how many actual clicks were successfully detected.
* **F1 Score:** The harmonic mean of precision and recall:
  $$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 24. Threshold Selection
The default classification threshold of 0.5 is not necessarily appropriate for an imbalanced CTR problem. Therefore, the validation set is used to find the threshold that produces the highest F1 score. The selected threshold is then applied to the test predictions.

* **DLRM Best Validation Threshold:** `0.08`

This threshold is selected using validation data rather than test data.

---

## 25. DLRM Results
The final DLRM test results are as follows:

| Metric | DLRM |
| :--- | :--- |
| **ROC-AUC** | 0.6674 |
| **PR-AUC** | 0.0688 |
| **Log Loss** | 0.1421 |
| **Accuracy** | 0.9312 |
| **F1** | 0.0777 |
| **Precision** | 0.0706 |
| **Recall** | 0.0866 |
| **Best Threshold** | 0.08 |
| **Parameters** | 2,900,513 |

---

## 26. Calibration
CTR models output probabilities, so it is important to examine whether these probabilities are meaningful. A calibration curve compares the **mean predicted probability** against the **actual fraction of positive samples**. Perfect calibration follows $y = x$.

The DLRM calibration curve is concentrated near the lower-probability region because the dataset contains relatively few positive click examples.

* **Brier Score:** `0.03207` (Lower Brier Score indicates better probabilistic accuracy)

The calibration curve should be interpreted together with other metrics rather than in isolation.

---

## 27. Ablation Study — Removing Interactions
To determine whether the explicit interaction mechanism actually contributes to performance, an ablation model was created.

The ablation removes the 351 explicit pairwise interaction features. Instead, it directly concatenates:
$$32 \text{ (dense)} + (26 \times 32) \text{ (embeddings)} = 864 \text{ dimensions}$$

```python
ablation_input = layers.Concatenate(
    name="ablation_input"
)([dense] + embedded_features)
```

The rest of the top MLP is kept identical. This creates a controlled experiment:

$$\text{\textbf{Full DLRM: }} \text{Dense} + \text{Explicit Interactions} + \text{Top MLP}$$
$$\text{vs.}$$
$$\text{\textbf{Ablation: }} \text{Dense} + \text{Embeddings} + \text{Top MLP}$$

---

## 28. Ablation Results

| Metric | DLRM | No Interactions |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.6674** | 0.6250 |
| **PR-AUC** | **0.0688** | 0.0635 |
| **Log Loss** | **0.1421** | 0.1567 |
| **Accuracy** | **0.9312** | 0.8881 |
| **F1** | 0.0777 | **0.1112** |
| **Precision** | 0.0706 | **0.0758** |
| **Recall** | 0.0866 | **0.2090** |

Removing explicit interactions causes a substantial drop in **ROC-AUC** and **PR-AUC**, alongside an increase in **Log Loss**. This provides empirical evidence that explicit feature interactions are useful for this CTR prediction task.

*(Note: The ablation happens to achieve higher F1 and recall at its selected threshold, so DLRM should not be described as better on every single threshold-based metric.)*

---

## 29. DLRM vs. Vanilla Neural Network

| Metric | Vanilla NN | DLRM |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.6794** | 0.6674 |
| **PR-AUC** | **0.0695** | 0.0688 |
| **Log Loss** | **0.1417** | 0.1421 |
| **Accuracy** | 0.9134 | **0.9312** |
| **F1** | **0.1035** | 0.0777 |
| **Precision** | **0.0792** | 0.0706 |
| **Recall** | **0.1493** | 0.0866 |

For this particular dataset and implementation, DLRM does not outperform the vanilla neural network on the main ranking metrics.
* The vanilla neural network has slightly better **ROC-AUC**, **PR-AUC**, **Log Loss**, and **F1**.
* DLRM has higher **Accuracy** (though accuracy is less informative for highly imbalanced CTR tasks).

---

## 30. Why DLRM Can Still Be Architecturally Superior
The DLRM architecture provides a more explicit way to represent feature interactions.

* **Vanilla Neural Network:**
  $$\text{Embeddings} + \text{Numerical Features} \rightarrow \text{Concatenation} \rightarrow \text{MLP}$$
* **DLRM:**
  $$\text{Embeddings} + \text{Numerical Representation} \rightarrow \text{Explicit Pairwise Interactions} \rightarrow \text{Top MLP}$$

DLRM introduces a stronger inductive bias toward learning useful feature combinations. The ablation experiment supports this:
* **DLRM ROC-AUC:** `0.6674`
* **Without Interactions ROC-AUC:** `0.6250`

This indicates that the interaction mechanism itself is useful, even though the complete DLRM did not outperform Task 2 on this specific dataset.

---

## 31. DLRM vs. Matrix Factorization
Task 1 uses Matrix Factorization, which models interactions primarily between **User** and **Item** using learned latent vectors.

DLRM is more general. Instead of limiting interactions to a user-item matrix, it combines:
* User information
* Advertisement information
* Campaign information
* Device information
* Context information
* Numerical signals

and explicitly models pairwise relationships between their learned representations.

**Architectural Evolution:**
$$\text{Matrix Factorization (User-Item Latent Interaction)} \rightarrow \text{Vanilla NN (Nonlinear Representation Learning)} \rightarrow \text{DLRM (Embeddings + Explicit Interactions + Top Network)}$$

---

## 32. Advantages of DLRM
1. **Handles categorical features naturally:** Each categorical field receives its own embedding table.
2. **Explicit feature interactions:** Pairwise interactions are directly calculated rather than relying entirely on the MLP to discover them implicitly.
3. **Supports heterogeneous features:** Numerical and categorical features can be processed differently.
4. **Purpose-built for recommendation/CTR:** The architecture directly mirrors the inherent structure of recommendation data.
5. **Explicit representations:** The model explicitly represents relationships between feature vectors.

---

## 33. Limitations and Costs
1. **More parameters:** Contains approximately 2.9 million parameters.
2. **Higher computational load:** With 27 feature vectors, the model explicitly calculates 351 pairwise dot products.
3. **Memory footprint:** Large categorical vocabularies produce substantial embedding tables.
4. **Architectural complexity:** Requires embedding tables + Bottom MLP + Interaction layer + Top MLP instead of a basic concatenation stack.
5. **Preprocessing overhead:** Each categorical field requires its own vocabulary mapping and embedding table.

---

## 34. Key Implementation Decisions Summary

| Component | Choice / Value |
| :--- | :--- |
| **Numerical fields** | 13 |
| **Categorical fields** | 26 |
| **Numerical imputation** | Median |
| **Numerical scaling** | StandardScaler |
| **Unknown category index** | 0 |
| **Embedding dimension** | 32 |
| **Bottom MLP** | 64 $\rightarrow$ 32 |
| **Number of feature vectors** | 27 |
| **Pairwise interactions** | 351 |
| **Interaction operation** | Dot product |
| **Interaction representation** | 351 scalars |
| **Combined representation** | 383 dimensions |
| **Top MLP** | 128 $\rightarrow$ 64 |
| **Dropout** | 0.2 |
| **Output** | Sigmoid |
| **Loss** | Binary cross-entropy |
| **Optimizer** | Adam |
| **Learning rate** | 0.001 |
| **Batch size** | 256 |
| **Maximum epochs** | 20 |
| **Early stopping monitor** | Validation PR-AUC |
| **Early stopping patience**| 3 |

---

## 35. Conclusion
The DLRM implementation successfully demonstrates how recommendation models combine dense numerical representations, categorical embeddings, explicit pairwise interactions, and deep nonlinear processing.

**Summary Results:**
* **ROC-AUC:** `0.6674`
* **PR-AUC:** `0.0688`
* **Log Loss:** `0.1421`

The ablation study confirmed the importance of explicit interactions (ROC-AUC dropped from `0.6674` to `0.6250` without them). However, on this dataset, DLRM did not outperform the simpler Task 2 vanilla neural network on primary ranking metrics. 

DLRM provides a tailored architecture for feature interaction learning, but as demonstrated, a more sophisticated architecture does not automatically guarantee superior performance on every dataset configuration.