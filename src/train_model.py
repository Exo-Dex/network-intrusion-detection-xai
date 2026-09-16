"""
Model definitions, architectures, and training procedures for supervised ML,
deep learning (MLP), and unsupervised anomaly detection (Autoencoder).
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    tf = None


def get_ml_model(name, is_multiclass=False, random_state=42):
    """
    Factory function returning initialized Scikit-learn / XGBoost classifiers.
    Handles scikit-learn parameter deprecations gracefully.
    """
    name = name.lower()
    if name in ['random_forest', 'rf']:
        return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif name in ['xgboost', 'xgb']:
        if is_multiclass:
            return xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=random_state, n_jobs=-1)
        else:
            return xgb.XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=random_state, n_jobs=-1)
    elif name in ['decision_tree', 'dt']:
        return DecisionTreeClassifier(max_depth=20, random_state=random_state)
    elif name in ['logistic_regression', 'lr']:
        # Note: newer scikit-learn versions deprecate multi_class; default handles binary and multiclass
        return LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model name: {name}")


def train_ml(model, X_train, y_train):
    """Fit a classical machine learning model."""
    model.fit(X_train, y_train)
    return model


def build_mlp(input_dim, num_classes=2):
    """
    Construct a 4-layer Multi-Layer Perceptron with BatchNormalization and Dropout.
    Binary task uses sigmoid; multiclass task uses softmax.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build deep learning models.")
    
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1 if num_classes == 2 else num_classes, 
              activation='sigmoid' if num_classes == 2 else 'softmax')
    ])
    
    loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model


def build_autoencoder(input_dim):
    """
    Construct a Deep Symmetric Autoencoder for unsupervised reconstruction-based
    anomaly detection (128 -> 64 -> Bottleneck(32) -> 64 -> 128 -> input_dim).
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build deep learning models.")
        
    inp = Input(shape=(input_dim,))
    # Encoder
    x = Dense(128, activation='relu')(inp)
    x = Dense(64, activation='relu')(x)
    bottleneck = Dense(32, activation='relu', name='bottleneck')(x)
    # Decoder
    x = Dense(64, activation='relu')(bottleneck)
    x = Dense(128, activation='relu')(x)
    out = Dense(input_dim, activation='linear')(x)
    
    autoencoder = Model(inputs=inp, outputs=out, name='nids_autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def train_dl(model, X_train, y_train=None, epochs=30, batch_size=128, val_split=0.1, patience=5):
    """Train a deep neural network with EarlyStopping on validation loss."""
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)]
    targets = X_train if y_train is None else y_train
    history = model.fit(
        X_train, targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1
    )
    return model, history


def compute_reconstruction_error(model, X):
    """Compute per-sample Mean Squared Error between input and reconstructed output."""
    preds = model.predict(X, verbose=0)
    mse = np.mean(np.square(X - preds), axis=1)
    return mse


def calibrate_autoencoder_threshold(model, X_normal_train, percentile=95.0):
    """Set anomaly detection threshold at the given percentile of normal reconstruction errors."""
    normal_errors = compute_reconstruction_error(model, X_normal_train)
    threshold = np.percentile(normal_errors, percentile)
    return threshold


def predict_autoencoder(model, X, threshold):
    """Flag samples with reconstruction error > threshold as attacks (1), else normal (0)."""
    errors = compute_reconstruction_error(model, X)
    preds = (errors > threshold).astype(int)
    return preds, errors
