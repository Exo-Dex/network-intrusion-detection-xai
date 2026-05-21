"""
train_model.py
--------------
Model definitions and training utilities for the NIDS-XAI project.

Models:
  ML  — Random Forest, XGBoost, Decision Tree, Logistic Regression
  DL  — MLP (Keras), Autoencoder (Keras, anomaly detection)
"""

import numpy as np
from sklearn.ensemble     import RandomForestClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost              import XGBClassifier


# ─────────────────────────────────────────
#  ML Models
# ─────────────────────────────────────────

def get_ml_model(name: str):
    """Return a scikit-learn / XGBoost model by name."""
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", use_label_encoder=False, n_jobs=-1
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=20, random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models.keys())}")
    return models[name]


def train_ml(model, X_train, y_train):
    """Fit an ML model and return it."""
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────
#  DL Models  (Keras / TensorFlow)
# ─────────────────────────────────────────

def build_mlp(input_dim: int, num_classes: int = 2):
    """
    Build a Multi-Layer Perceptron for classification.

    Args:
        input_dim  : number of input features
        num_classes: 2 for binary, >2 for multiclass

    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    activation = "sigmoid" if num_classes == 2 else "softmax"
    loss       = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"
    output_dim = 1 if num_classes == 2 else num_classes

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(output_dim, activation=activation),
    ], name="MLP")

    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=["accuracy"]
    )
    return model


def build_autoencoder(input_dim: int):
    """
    Build an Autoencoder for anomaly detection.
    Trained on NORMAL traffic only.
    High reconstruction error → likely an attack.

    Args:
        input_dim: number of input features

    Returns:
        autoencoder (full model), encoder (encoder half)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Encoder
    inputs  = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation="relu")(inputs)
    encoded = layers.Dense(64,  activation="relu")(encoded)
    encoded = layers.Dense(32,  activation="relu")(encoded)

    # Decoder
    decoded = layers.Dense(64,  activation="relu")(encoded)
    decoded = layers.Dense(128, activation="relu")(decoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = models.Model(inputs, decoded, name="Autoencoder")
    encoder     = models.Model(inputs, encoded, name="Encoder")

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def train_dl(model, X_train, y_train=None,
             epochs=30, batch_size=256, validation_split=0.1,
             verbose=1):
    """
    Train a Keras model.
    For autoencoder: pass X_train as both input and target (y_train=None).
    """
    import tensorflow as tf

    y = X_train if y_train is None else y_train   # autoencoder reconstructs input
    history = model.fit(
        X_train, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]
    )
    return history


def get_autoencoder_threshold(autoencoder, X_normal, percentile=95):
    """
    Compute reconstruction-error threshold from normal training samples.
    Samples above this threshold are flagged as anomalies (attacks).
    """
    recon       = autoencoder.predict(X_normal, verbose=0)
    errors      = np.mean(np.power(X_normal - recon, 2), axis=1)
    threshold   = np.percentile(errors, percentile)
    return threshold, errors


def autoencoder_predict(autoencoder, X, threshold):
    """
    Predict binary labels using reconstruction error.
    0 = Normal, 1 = Attack (anomaly)
    """
    recon  = autoencoder.predict(X, verbose=0)
    errors = np.mean(np.power(X - recon, 2), axis=1)
    preds  = (errors > threshold).astype(int)
    return preds, errors
