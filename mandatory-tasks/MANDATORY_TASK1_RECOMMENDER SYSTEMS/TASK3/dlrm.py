import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, Model

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

#identify features
numeric_cols = [
    col for col in train.columns
    if col.startswith("integer_")
]

categorical_cols = [
    col for col in train.columns
    if col.startswith("categorical_")
]

print("Number of numerical features:", len(numeric_cols))
print("Number of categorical features:", len(categorical_cols))

print("Numerical columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

X = train.drop("label", axis=1)
y = train["label"]

print("Target distribution:")
print(y.value_counts())

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])

#numerical preprocessing
numeric_imputer = SimpleImputer(strategy="median")
numeric_scaler = StandardScaler()

X_train_numeric = numeric_imputer.fit_transform(
    X_train[numeric_cols]
)

X_val_numeric = numeric_imputer.transform(
    X_val[numeric_cols]
)

X_train_numeric = numeric_scaler.fit_transform(
    X_train_numeric
)

X_val_numeric = numeric_scaler.transform(
    X_val_numeric
)

print("Training numerical shape:", X_train_numeric.shape)
print("Validation numerical shape:", X_val_numeric.shape)


#categorical preprocessing
categorical_mappings = {}

X_train_categorical = []
X_val_categorical = []

for col in categorical_cols:

    train_values = (
        X_train[col]
        .fillna("MISSING")
        .astype(str)
    )

    val_values = (
        X_val[col]
        .fillna("MISSING")
        .astype(str)
    )

    categories = train_values.unique()

    mapping = {
        category: index + 1
        for index, category in enumerate(categories)
    }

    categorical_mappings[col] = mapping

    train_encoded = (
        train_values
        .map(mapping)
        .fillna(0)
        .astype(int)
    )

    val_encoded = (
        val_values
        .map(mapping)
        .fillna(0)
        .astype(int)
    )

    X_train_categorical.append(train_encoded.values)
    X_val_categorical.append(val_encoded.values)

X_train_categorical = np.column_stack(
    X_train_categorical
)

X_val_categorical = np.column_stack(
    X_val_categorical
)

X_train_categorical = X_train_categorical.astype(np.int32)
X_val_categorical = X_val_categorical.astype(np.int32)

print("Training categorical shape:",
      X_train_categorical.shape)

print("Validation categorical shape:",
      X_val_categorical.shape)


#BUILD DRLM
numeric_input = layers.Input(
    shape=(len(numeric_cols),),
    name="numeric_input"
)

print("Numerical input shape:", numeric_input.shape)

categorical_input = layers.Input(
    shape=(len(categorical_cols),),
    dtype="int32",
    name="categorical_input"
)

print("Categorical input shape:", categorical_input.shape)
#bottom MLP
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

print("Dense representation shape:", dense.shape)

#Categorical embeddings
embedding_dim = 32

embedded_features = []

for i, col in enumerate(categorical_cols):

    vocabulary_size = int(
        X_train_categorical[:, i].max()
    ) + 1

    embedding = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        name=f"dlrm_embedding_{i}"
    )(categorical_input[:, i])

    embedded_features.append(embedding)

print(
    "Number of embedding features:",
    len(embedded_features)
)

print("Dense representation:", dense.shape)

for i, embedding in enumerate(embedded_features):
    print(
        f"Embedding {i + 1}:",
        embedding.shape
    )

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

print("Number of pairwise interactions:",
      len(interaction_features))


#Build DLRM input
interaction_vector = layers.Concatenate(
    name="interaction_vector"
)(interaction_features)

dlrm_input = layers.Concatenate(
    name="dlrm_input"
)([
    dense,
    interaction_vector
])

print("Interaction vector shape:", interaction_vector.shape)
print("DLRM input shape:", dlrm_input.shape)

#top_MlP
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
dlrm_model = Model(
    inputs=[numeric_input, categorical_input],
    outputs=output,
    name="DLRM"
)

dlrm_model.summary()

#Compliling the DLRM
dlrm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy")
    ]
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)

#train
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
test = pd.read_csv("test.csv")

X_test = test.drop("label", axis=1)
y_test = test["label"]

# Numerical test preprocessing
X_test_numeric = numeric_imputer.transform(
    X_test[numeric_cols]
)

X_test_numeric = numeric_scaler.transform(
    X_test_numeric
)
# Categorical test preprocessing
X_test_categorical_list = []

for col in categorical_cols:

    train_values = X_train[col].fillna("MISSING").astype(str)
    test_values = X_test[col].fillna("MISSING").astype(str)

    categories = train_values.unique()

    category_to_index = {
        category: index + 1
        for index, category in enumerate(categories)
    }

    encoded_test = test_values.map(
        lambda x: category_to_index.get(x, 0)
    )

    X_test_categorical_list.append(encoded_test.values)

X_test_categorical = np.column_stack(
    X_test_categorical_list
).astype("int32")

print("Test numerical shape:", X_test_numeric.shape)
print("Test categorical shape:", X_test_categorical.shape)
print("Test labels shape:", y_test.shape)

#Evaluation
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# Predictions
val_pred = dlrm_model.predict(
    [X_val_numeric, X_val_categorical],
    verbose=0
).ravel()

test_pred = dlrm_model.predict(
    [X_test_numeric, X_test_categorical],
    verbose=0
).ravel()

# Basic metrics
print("Validation ROC-AUC:",
      roc_auc_score(y_val, val_pred))

print("Validation PR-AUC:",
      average_precision_score(y_val, val_pred))

print("Validation Log Loss:",
      log_loss(y_val, val_pred))

thresholds = np.arange(0.05, 0.96, 0.01)

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    val_labels = (val_pred >= threshold).astype(int)

    current_f1 = f1_score(y_val, val_labels)

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold

print("Best threshold:", best_threshold)
print("Best validation F1:", best_f1)

test_labels = (test_pred >= best_threshold).astype(int)

print("Test ROC-AUC:",
      roc_auc_score(y_test, test_pred))

print("Test PR-AUC:",
      average_precision_score(y_test, test_pred))

print("Test Log Loss:",
      log_loss(y_test, test_pred))

print("Test Accuracy:",
      accuracy_score(y_test, test_labels))

print("Test F1:",
      f1_score(y_test, test_labels))

print("Test Precision:",
      precision_score(y_test, test_labels))

print("Test Recall:",
      recall_score(y_test, test_labels))


#ablation code
ablation_input = dense

ablation_top = layers.Dense(
    128,
    activation="relu",
    name="ablation_top_dense_1"
)(ablation_input)

ablation_top = layers.Dropout(
    0.2,
    name="ablation_top_dropout"
)(ablation_top)

ablation_top = layers.Dense(
    64,
    activation="relu",
    name="ablation_top_dense_2"
)(ablation_top)

ablation_output = layers.Dense(
    1,
    activation="sigmoid",
    name="ablation_click_probability"
)(ablation_top)

ablation_model = Model(
    inputs=[numeric_input, categorical_input],
    outputs=ablation_output,
    name="DLRM_Ablation_NoInteractions"
)

ablation_model.summary()

ablation_input = layers.Concatenate(
    name="ablation_input"
)([dense] + embedded_features)

print("Ablation input shape:", ablation_input.shape)

#ablation mlp
ablation_top = layers.Dense(
    128,
    activation="relu",
    name="ablation_top_dense_1"
)(ablation_input)

ablation_top = layers.Dropout(
    0.2,
    name="ablation_top_dropout"
)(ablation_top)

ablation_top = layers.Dense(
    64,
    activation="relu",
    name="ablation_top_dense_2"
)(ablation_top)

ablation_output = layers.Dense(
    1,
    activation="sigmoid",
    name="ablation_click_probability"
)(ablation_top)

ablation_model = Model(
    inputs=[numeric_input, categorical_input],
    outputs=ablation_output,
    name="DLRM_Ablation_NoInteractions"
)

ablation_model.summary()

#training the ablation
ablation_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy")
    ]
)
ablation_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)

ablation_history = ablation_model.fit(
    [X_train_numeric, X_train_categorical],
    y_train,
    validation_data=(
        [X_val_numeric, X_val_categorical],
        y_val
    ),
    epochs=20,
    batch_size=256,
    callbacks=[ablation_early_stopping],
    verbose=1
)

ablation_val_pred = ablation_model.predict(
    [X_val_numeric, X_val_categorical],
    verbose=0
).ravel()

ablation_test_pred = ablation_model.predict(
    [X_test_numeric, X_test_categorical],
    verbose=0
).ravel()

print("Validation ROC-AUC:",
      roc_auc_score(y_val, ablation_val_pred))

print("Validation PR-AUC:",
      average_precision_score(y_val, ablation_val_pred))

print("Validation Log Loss:",
      log_loss(y_val, ablation_val_pred))

ablation_thresholds = np.arange(0.05, 0.96, 0.01)

best_ablation_threshold = 0
best_ablation_f1 = 0

for threshold in ablation_thresholds:

    val_labels = (
        ablation_val_pred >= threshold
    ).astype(int)

    current_f1 = f1_score(y_val, val_labels)

    if current_f1 > best_ablation_f1:
        best_ablation_f1 = current_f1
        best_ablation_threshold = threshold

print("Best ablation threshold:",
      best_ablation_threshold)

print("Best validation F1:",
      best_ablation_f1)

ablation_test_labels = (
    ablation_test_pred >= best_ablation_threshold
).astype(int)

print("Test ROC-AUC:",
      roc_auc_score(y_test, ablation_test_pred))

print("Test PR-AUC:",
      average_precision_score(y_test, ablation_test_pred))

print("Test Log Loss:",
      log_loss(y_test, ablation_test_pred))

print("Test Accuracy:",
      accuracy_score(y_test, ablation_test_labels))

print("Test F1:",
      f1_score(y_test, ablation_test_labels))

print("Test Precision:",
      precision_score(y_test, ablation_test_labels))

print("Test Recall:",
      recall_score(y_test, ablation_test_labels))

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(
    y_test,
    test_pred,
    n_bins=10,
    strategy="uniform"
)

plt.figure(figsize=(7, 7))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="DLRM"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("DLRM Calibration Curve")
plt.legend()
plt.grid()

plt.show()

from sklearn.metrics import brier_score_loss

dlrm_brier = brier_score_loss(y_test, test_pred)

print("DLRM Brier Score:", dlrm_brier)