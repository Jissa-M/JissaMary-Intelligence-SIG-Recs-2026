# Task 2 — Neural CTR Prediction

## 1. Objective

The objective of this task is to build a neural-network baseline for **Click-Through Rate (CTR) prediction**.

The dataset contains numerical and categorical features representing user, item, and contextual information. The target variable is a binary `label`:

- `0` → no click
- `1` → click

The task focuses on:

- Preprocessing numerical and categorical features
- Learning embeddings for categorical variables
- Training multiple neural-network architectures
- Selecting the best architecture using validation data
- Investigating overfitting
- Evaluating the model using multiple classification and ranking metrics
- Selecting a suitable classification threshold
- Analysing model calibration

A **Deep & Cross Network (DCN)** is an optional bonus extension.

---

# 2. Dataset

The dataset contains separate training and test files.

| Dataset | Rows | Columns |
|---|---:|---:|
| Training | 40,000 | 40 |
| Test | 10,000 | 40 |

The columns consist of:

- `label`
- 13 numerical features:
  - `integer_feature_1` to `integer_feature_13`
- 26 categorical features:
  - `categorical_feature_1` to `categorical_feature_26`

The training data contains:

- **38,720 negative examples**
- **1,280 positive examples**

Therefore, the positive class represents approximately **3.2%** of the training data.

This makes the dataset highly imbalanced, which is important when interpreting accuracy and other evaluation metrics.

---

# 3. Train / Validation / Test Split

The provided training data was split into:

- **80% training**
- **20% validation**

The split was stratified so that the positive-class proportion remained approximately the same in both subsets.

```python
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This resulted in:

- Training: 32,000 samples
- Validation: 8,000 samples

The separate test set was kept untouched during model selection.

This is important because the test set should only be used for the final evaluation of the selected model.

---

# 4. Preprocessing

## 4.1 Numerical Features

The numerical features contain missing values.

Two preprocessing steps were applied.

### Median Imputation

Missing numerical values were replaced using the median of the corresponding feature.

```python
numeric_imputer = SimpleImputer(strategy="median")
```

The imputer was fitted only on the training data.

### Standardization

The numerical features were then standardized:

$$
z = \frac{x-\mu}{\sigma}
$$

where:

- $x$ = original feature value
- $\mu$ = training-set mean
- $\sigma$ = training-set standard deviation

```python
numeric_scaler = StandardScaler()
```

The same fitted imputer and scaler were then applied to validation and test data.

This prevents information from the validation or test sets from leaking into preprocessing.

---

# 5. Categorical Features

The categorical features contain both low-cardinality and high-cardinality variables.

Each categorical feature was converted into integer IDs.

Missing values were represented using:

```text
"MISSING"
```

A separate mapping was created using the training data.

The encoding reserves:

```text
0 → unknown / unseen category
1, 2, 3, ... → known categories
```

For validation and test data, categories that were not present during training were mapped to `0`.

The categorical values were then converted into integer tensors for use by embedding layers.

---

# 6. Categorical Embeddings

Instead of using one-hot encoding, the neural network uses **embedding layers** for categorical variables.

An embedding converts a categorical ID into a learned dense vector.

```text
Category ID
     ↓
Embedding Layer
     ↓
Dense vector
```

The embedding dimensions were selected based on vocabulary size, with a maximum embedding dimension of 50.

The embedding vectors are trainable parameters, meaning the network learns representations of categories during training.

Embeddings are particularly useful for high-cardinality categorical features because they provide compact representations instead of creating extremely large one-hot vectors.

---

# 7. Model Architecture

Three small neural-network architectures were tested.

The models use:

1. Numerical features
2. Categorical embeddings
3. Concatenation of all features
4. Dense neural-network layers
5. Dropout
6. A sigmoid output layer

The final sigmoid output represents the predicted probability of a click.

The sigmoid function is:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

---

# 8. Model 1

Model 1 used:

```text
Categorical embeddings
        +
Numerical features
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.2)
        ↓
Dense(64, ReLU)
        ↓
Dropout(0.2)
        ↓
Dense(1, Sigmoid)
```

Model 1 contained approximately **4.57 million trainable parameters**.

The model achieved strong training performance but showed a large difference between training and validation performance.

---

# 9. Model 2

Model 2 reduced the dense-network capacity and increased dropout:

```text
Categorical embeddings
        +
Numerical features
        ↓
Dense(64, ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(32, ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(1, Sigmoid)
```

The purpose was to reduce overfitting and improve generalization.

Validation performance improved slightly compared with Model 1.

---

# 10. Model 3

Model 3 further reduced the embedding capacity while using the smaller dense architecture.

```text
Categorical embeddings
        +
Numerical features
        ↓
Dense(64, ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(32, ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(1, Sigmoid)
```

Model 3 was selected as the best architecture based on validation performance.

---

# 11. Training

The models were trained using:

- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Output activation:** Sigmoid
- **Early stopping:** Used to monitor validation performance
- **Validation metric:** PR-AUC was given particular importance because of the strong class imbalance

Binary cross-entropy is:

$$
L=-[y\log(p)+(1-y)\log(1-p)]
$$

where:

- $y$ = true label
- $p$ = predicted probability

---

# 12. Overfitting Analysis

Training and validation curves were plotted to investigate overfitting.

For Model 3, the training PR-AUC increased strongly over the epochs, reaching approximately 0.79.

However, validation PR-AUC decreased after its early peak.

Similarly:

- Training loss continuously decreased.
- Validation loss initially remained relatively low.
- Validation loss then increased while training loss continued decreasing.

This is a clear indication of **overfitting**.

The model therefore learned patterns that were increasingly specific to the training data and did not generalize equally well to the validation data.

Early stopping was used to restore the best validation model rather than using the final overfit epoch.

---

# 13. Validation Results

The best validation evaluation for Model 3 was:

| Metric | Validation Result |
|---|---:|
| ROC-AUC | 0.7102 |
| PR-AUC | 0.0865 |
| Accuracy | 0.9680 |
| Log Loss | 0.1330 |
| F1 | 0.1619 |
| Precision | 0.1093 |
| Recall | 0.3125 |

Because the dataset is highly imbalanced, PR-AUC, ROC-AUC, and threshold-based metrics are more informative than accuracy alone.

---

# 14. Classification Threshold

The neural network produces probabilities rather than direct class labels.

For example:

```text
0.02
0.04
0.08
0.31
```

A threshold is required to convert these probabilities into class predictions.

Instead of automatically using the conventional threshold of `0.5`, the threshold was selected using the validation set to improve F1.

The selected threshold was approximately **0.06–0.07**, depending on the validation run.

The important principle is that the threshold was determined using validation data and was then kept fixed for test evaluation.

---

# 15. Why Not Use Accuracy Alone?

The dataset contains only about 3.2% positive examples.

A model that predicts almost everything as `0` could achieve high accuracy while being poor at detecting clicks.

Therefore, accuracy alone is not sufficient.

Important metrics include:

### Precision

$$
Precision=\frac{TP}{TP+FP}
$$

Precision measures how many predicted positives were actually positive.

### Recall

$$
Recall=\frac{TP}{TP+FN}
$$

Recall measures how many actual positive examples were detected.

### F1 Score

$$
F1=2\frac{Precision\times Recall}{Precision+Recall}
$$

F1 balances precision and recall.

---

# 16. ROC-AUC

ROC-AUC measures how well the model ranks positive examples above negative examples across different thresholds.

A value of:

```text
0.5 → approximately random ranking
1.0 → perfect ranking
```

Model 3 achieved:

```text
Validation ROC-AUC = 0.7102
Test ROC-AUC       = 0.6725
```

The decrease from validation to test indicates some generalization gap, but the test performance remains above random ranking.

---

# 17. PR-AUC

PR-AUC summarizes the relationship between precision and recall across different thresholds.

It is especially useful for this task because the positive class is rare.

Model 3 achieved:

```text
Validation PR-AUC = 0.0865
Test PR-AUC       = 0.0703
```

The positive-class prevalence is approximately:

```text
3.2%
```

Therefore, the PR-AUC is evaluated relative to a difficult imbalanced classification setting.

---

# 18. Final Test Results

The selected Model 3 was evaluated on the untouched test set.

| Metric | Test Result |
|---|---:|
| **ROC-AUC** | **0.6725** |
| **PR-AUC** | **0.0703** |
| **Accuracy** | **0.8902** |
| **Log Loss** | **0.1429** |
| **F1** | **0.1300** |
| **Precision** | **0.0885** |
| **Recall** | **0.2448** |

The test threshold used was the threshold selected during validation.

---

# 19. Calibration Analysis

A calibration / reliability plot was produced to compare:

```text
Mean predicted probability
```

with:

```text
Observed fraction of positives
```

A perfectly calibrated model would follow the diagonal:

$$
y=x
$$

The quantile-bin calibration plot showed that Model 3's predictions were reasonably close to the ideal calibration line in the low-probability region, although the model was not perfectly calibrated.

Quantile bins were useful because most predicted CTR probabilities were concentrated near zero.

---

# 20. Architecture Comparison

| Model | Dense Architecture | Dropout | Main Observation |
|---|---|---|---|
| Model 1 | 128 → 64 | 0.2 | Higher capacity, stronger overfitting |
| Model 2 | 64 → 32 | 0.3 | Reduced overfitting, slight validation improvement |
| Model 3 | 64 → 32 | 0.3 + smaller embeddings | Best validation performance |

Model 3 was selected because it provided the strongest validation performance among the tested architectures while using a smaller representation.

The test set was not used to select the architecture.

---

# 21. Key Findings

The main findings from the experiment are:

1. Categorical embeddings provide a practical way to represent high-cardinality categorical features.
2. Numerical features require appropriate imputation and scaling.
3. Validation data is essential for architecture selection and threshold tuning.
4. Accuracy can be misleading in highly imbalanced CTR prediction.
5. PR-AUC is particularly useful when positive examples are rare.
6. Increasing model capacity can cause substantial overfitting.
7. Dropout and reduced model capacity can improve generalization.
8. The final test performance was lower than validation performance, showing a generalization gap.
9. The predicted probabilities were reasonably calibrated in the main low-probability region but were not perfectly calibrated.
10. Model selection and threshold selection must be completed without using the test set.

---

# 22. Limitations

The current baseline has several limitations:

- The neural network still shows noticeable overfitting.
- The dataset is strongly imbalanced.
- The model's probability calibration is not perfect.
- Only a small number of architectures were tested.
- No extensive hyperparameter search was performed.
- The vanilla network does not explicitly model feature crosses.
- The optional DCN extension has not yet been included.

---

# 23. Bonus: Deep & Cross Network

A **Deep & Cross Network (DCN)** can be used as an extension to the vanilla neural network.

The motivation is to explicitly model feature interactions.

For example, CTR may depend on combinations such as:

```text
user × device
user × placement
item × device
campaign × placement
```

A standard deep network can learn nonlinear interactions implicitly.

A DCN adds explicit cross layers to learn such interactions directly.

The DCN experiment can therefore answer:

> Does explicitly learning feature crosses improve CTR prediction compared with the vanilla neural-network baseline?

The DCN paper used as the reference is:

**Deep & Cross Network for Ads Prediction**

https://arxiv.org/abs/1708.05123

---

# 24. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib

---

# 25. Project Structure

A possible project structure is:

```text
TASK2/
│
├── Neural_ctr.py
├── README.md
│
├── datasets/
│   └── ctr_dataset/
│       ├── train.csv
│       └── test.csv
│
└── plots/
    ├── model3_pr_auc.png
    ├── model3_loss.png
    └── calibration.png
```

---

# 26. Conclusion

A vanilla neural CTR prediction model was developed using numerical features and learned categorical embeddings.

Three neural-network architectures were tested using a separate validation set. Model 3 achieved the best validation performance and was selected for final evaluation.

The final test performance was:

- **ROC-AUC: 0.6725**
- **PR-AUC: 0.0703**
- **Log Loss: 0.1429**
- **F1: 0.1300**

The training curves demonstrated clear overfitting, highlighting the importance of validation monitoring and regularization.

The project also demonstrated why CTR prediction requires more than accuracy: the strong class imbalance makes ranking metrics such as ROC-AUC and PR-AUC, along with precision, recall, F1, log loss, and calibration, important for evaluating the model properly.

The next possible extension is a **Deep & Cross Network (DCN)** to investigate whether explicit feature interactions can improve upon the vanilla neural-network baseline.
