import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain columns:")
print(train.columns.tolist())

print("\nTarget distribution:")
print(train["label"].value_counts())

# Numerical and categorical columns

numeric_cols = [col for col in train.columns if col.startswith("integer_")]
categorical_cols = [col for col in train.columns if col.startswith("categorical_")]

print("Number of numerical features:", len(numeric_cols))
print("Number of categorical features:", len(categorical_cols))

print("\nMissing values:")
print(train.isnull().sum())

print("\nUnique values in categorical features:")
print(train[categorical_cols].nunique())

#split the data
from sklearn.model_selection import train_test_split

X = train.drop("label", axis=1)
y = train["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nValidation target distribution:")
print(y_val.value_counts())

#Numverical preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Numerical preprocessing
numeric_imputer = SimpleImputer(strategy="median")
numeric_scaler = StandardScaler()

X_train_numeric = numeric_imputer.fit_transform(X_train[numeric_cols])
X_val_numeric = numeric_imputer.transform(X_val[numeric_cols])

X_train_numeric = numeric_scaler.fit_transform(X_train_numeric)
X_val_numeric = numeric_scaler.transform(X_val_numeric)

print("Training numerical shape:", X_train_numeric.shape)
print("Validation numerical shape:", X_val_numeric.shape)

print("\nMissing values after preprocessing:")
print(np.isnan(X_train_numeric).sum())

#Encode categorical features
# Encode categorical features

categorical_mappings = {}

X_train_categorical = []
X_val_categorical = []

for col in categorical_cols:

    # Convert to string and fill missing values
    train_values = X_train[col].fillna("MISSING").astype(str)
    val_values = X_val[col].fillna("MISSING").astype(str)

    # Get categories from TRAINING data only
    categories = train_values.unique()

    # Create mapping
    mapping = {category: index + 1 for index, category in enumerate(categories)}
    categorical_mappings[col] = mapping

    # Encode training data
    train_encoded = train_values.map(mapping).fillna(0).astype(int)

    # Encode validation data
    val_encoded = val_values.map(mapping).fillna(0).astype(int)

    X_train_categorical.append(train_encoded.values)
    X_val_categorical.append(val_encoded.values)

# Convert to arrays
X_train_categorical = np.column_stack(X_train_categorical)
X_val_categorical = np.column_stack(X_val_categorical)

print("Training categorical shape:", X_train_categorical.shape)
print("Validation categorical shape:", X_val_categorical.shape)

print("\nMaximum encoded value for each feature:")

for i, col in enumerate(categorical_cols):
    print(col, ":", X_train_categorical[:, i].max())
    
    
import tensorflow as tf
from tensorflow.keras import layers, Model
  
    # Numerical input
numeric_input = layers.Input(
    shape=(len(numeric_cols),),
    name="numeric_input"
)

# Categorical input
categorical_input = layers.Input(
    shape=(len(categorical_cols),),
    dtype="int32",
    name="categorical_input"
)

print("Numerical input shape:", numeric_input.shape)
print("Categorical input shape:", categorical_input.shape)

embedded_features = []

for i, col in enumerate(categorical_cols):

    vocabulary_size = int(X_train_categorical[:, i].max()) + 1

    # Choose a small embedding dimension
    embedding_dim = int(min(50, max(4, vocabulary_size // 2)))

    embedding = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        name=f"embedding_{i}"
    )(categorical_input[:, i])

    embedded_features.append(embedding)
print("All embedding layers created successfully.")
    
# Combine all categorical embeddings
categorical_combined = layers.Concatenate(
    name="categorical_combined"
)(embedded_features)

# Combine categorical embeddings with numerical features
combined = layers.Concatenate(
    name="combined_features"
)([categorical_combined, numeric_input])

print("Combined feature representation created.")

x = layers.Dense(128, activation="relu")(combined)
x = layers.Dropout(0.2)(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)

output = layers.Dense(1, activation="sigmoid")(x)

model = Model(
    inputs=[numeric_input, categorical_input],
    outputs=output
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        ),
        tf.keras.metrics.BinaryAccuracy(name="accuracy")
    ]
)

print("Model compiled successfully.")

#train model1
X_train_categorical = X_train_categorical.astype(np.int32)
X_val_categorical = X_val_categorical.astype(np.int32)

print(X_train_categorical.dtype)
print(X_val_categorical.dtype)

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
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
#training vs validation curves
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history["pr_auc"], label="Training PR-AUC")
plt.plot(history.history["val_pr_auc"], label="Validation PR-AUC")
plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.title("Training vs Validation PR-AUC")
plt.legend()
plt.show()

#Model 2
model2_numeric_input = layers.Input(
    shape=(len(numeric_cols),),
    name="numeric_input"
)

model2_categorical_input = layers.Input(
    shape=(len(categorical_cols),),
    dtype="int32",
    name="categorical_input"
)

model2_embedded_features = []

for i, col in enumerate(categorical_cols):
    vocabulary_size = int(X_train_categorical[:, i].max()) + 1
    embedding_dim = int(min(50, max(4, vocabulary_size // 2)))

    embedding = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        name=f"model2_embedding_{i}"
    )(model2_categorical_input[:, i])

    model2_embedded_features.append(embedding)

model2_categorical_combined = layers.Concatenate(
    name="model2_categorical_combined"
)(model2_embedded_features)

model2_combined = layers.Concatenate(
    name="model2_combined_features"
)([
    model2_categorical_combined,
    model2_numeric_input
])

x = layers.Dense(64, activation="relu")(model2_combined)
x = layers.Dropout(0.3)(x)

x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.3)(x)

model2_output = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model2 = Model(
    inputs=[model2_numeric_input, model2_categorical_input],
    outputs=model2_output
)

model2.summary()

# compile
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy")
    ]
)

print("Model 2 compiled successfully.")

#training
early_stopping2 = EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)

history2 = model2.fit(
    [X_train_numeric, X_train_categorical],
    y_train,
    validation_data=(
        [X_val_numeric, X_val_categorical],
        y_val
    ),
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping2],
    verbose=1
)

#model 3
# ============================================================
# MODEL 3 - SMALLER EMBEDDINGS
# ============================================================

model3_numeric_input = layers.Input(
    shape=(len(numeric_cols),),
    name="numeric_input"
)

model3_categorical_input = layers.Input(
    shape=(len(categorical_cols),),
    dtype="int32",
    name="categorical_input"
)

model3_embedded_features = []

for i, col in enumerate(categorical_cols):

    vocabulary_size = int(X_train_categorical[:, i].max()) + 1

    embedding_dim = int(
        min(16, max(4, vocabulary_size // 4))
    )

    embedding = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        name=f"model3_embedding_{i}"
    )(model3_categorical_input[:, i])

    model3_embedded_features.append(embedding)


model3_categorical_combined = layers.Concatenate(
    name="model3_categorical_combined"
)(model3_embedded_features)


model3_combined = layers.Concatenate(
    name="model3_combined_features"
)([
    model3_categorical_combined,
    model3_numeric_input
])


x = layers.Dense(64, activation="relu")(model3_combined)
x = layers.Dropout(0.3)(x)

x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.3)(x)

model3_output = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model3 = Model(
    inputs=[
        model3_numeric_input,
        model3_categorical_input
    ],
    outputs=model3_output
)

model3.summary()

#compile
model3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        ),
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        )
    ]
)

print("Model 3 compiled successfully.")

#train 
early_stopping3 = EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)

history3 = model3.fit(
    [X_train_numeric, X_train_categorical],
    y_train,
    validation_data=(
        [X_val_numeric, X_val_categorical],
        y_val
    ),
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping3],
    verbose=1
)
y_val_prob = model3.predict(
    [X_val_numeric, X_val_categorical],
    verbose=0
).ravel()

print(y_val_prob[:10])

#metrics
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    log_loss
)

val_roc_auc = roc_auc_score(y_val, y_val_prob)
val_pr_auc = average_precision_score(y_val, y_val_prob)
val_log_loss = log_loss(y_val, y_val_prob)

y_val_pred = (y_val_prob >= 0.5).astype(int)
val_accuracy = accuracy_score(y_val, y_val_pred)

print("Validation ROC-AUC:", val_roc_auc)
print("Validation PR-AUC:", val_pr_auc)
print("Validation Accuracy:", val_accuracy)
print("Validation Log Loss:", val_log_loss)

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

thresholds = np.arange(0.05, 0.96, 0.01)

best_threshold = 0
best_f1 = 0

for threshold in thresholds:
    y_val_pred = (y_val_prob >= threshold).astype(int)

    f1 = f1_score(y_val, y_val_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

y_val_pred = (y_val_prob >= best_threshold).astype(int)

val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)

print("Best threshold:", best_threshold)
print("Validation F1:", best_f1)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)

#calibration plot
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(
    y_val,
    y_val_prob,
    n_bins=10,
    strategy="uniform"
)

plt.figure(figsize=(7, 6))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Model 3"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration / Reliability Plot")
plt.legend()
plt.show()

#calibration plot 2
prob_true_q, prob_pred_q = calibration_curve(
    y_val,
    y_val_prob,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(7, 6))

plt.plot(
    prob_pred_q,
    prob_true_q,
    marker="o",
    label="Model 3"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration / Reliability Plot - Quantile Bins")
plt.legend()
plt.show()


#test-set preprocessing
# ============================================================
# TEST SET PREPROCESSING
# ============================================================

X_test = test.drop("label", axis=1)
y_test = test["label"]

X_test_numeric = numeric_imputer.transform(
    X_test[numeric_cols]
)

X_test_numeric = numeric_scaler.transform(
    X_test_numeric
)

print("Test numerical shape:", X_test_numeric.shape)

X_test_categorical = []

for col in categorical_cols:

    test_values = X_test[col].fillna("MISSING").astype(str)

    mapping = categorical_mappings[col]

    test_encoded = test_values.map(mapping).fillna(0).astype(int)

    X_test_categorical.append(test_encoded.values)

X_test_categorical = np.column_stack(
    X_test_categorical
)

X_test_categorical = X_test_categorical.astype(np.int32)

print("Test categorical shape:", X_test_categorical.shape)


#test predictions
y_test_prob = model3.predict(
    [X_test_numeric, X_test_categorical],
    verbose=0
).ravel()

print(y_test_prob[:10])
#final test metrics
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score
)

test_roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)

test_pr_auc = average_precision_score(
    y_test,
    y_test_prob
)

test_log_loss = log_loss(
    y_test,
    y_test_prob
)

# Use the threshold selected ONLY from validation
test_threshold = best_threshold

y_test_pred = (
    y_test_prob >= test_threshold
).astype(int)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_f1 = f1_score(
    y_test,
    y_test_pred
)

test_precision = precision_score(
    y_test,
    y_test_pred
)

test_recall = recall_score(
    y_test,
    y_test_pred
)

print("================================")
print("FINAL TEST RESULTS")
print("================================")
print("ROC-AUC:", test_roc_auc)
print("PR-AUC:", test_pr_auc)
print("Accuracy:", test_accuracy)
print("Log Loss:", test_log_loss)
print("Threshold:", test_threshold)
print("F1:", test_f1)
print("Precision:", test_precision)
print("Recall:", test_recall)

# ============================================================
# MODEL 3 - TRAINING CURVES
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(
    history3.history["pr_auc"],
    label="Training PR-AUC"
)

plt.plot(
    history3.history["val_pr_auc"],
    label="Validation PR-AUC"
)

plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.title("Model 3 - Training vs Validation PR-AUC")
plt.legend()
plt.show()


#loss curve


plt.figure(figsize=(8, 5))

plt.plot(
    history3.history["loss"],
    label="Training Loss"
)

plt.plot(
    history3.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Model 3 - Training vs Validation Loss")
plt.legend()
plt.show()
