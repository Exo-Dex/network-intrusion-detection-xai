"""
Temporal sequence construction and sliding-window generation for network flows.
Designed for authentic chronological flow streams (e.g. CIC-IDS2017) and host-aggregated
burst sequences (e.g. UNSW-NB15 / NSL-KDD destination host windowing).
"""

import numpy as np
import pandas as pd


def build_host_temporal_sequences(
    df,
    feature_cols,
    target_col='binary_label',
    host_col=None,
    time_col=None,
    window_size=10,
    step_size=1,
    padding='pre'
):
    """
    Construct sequential flow windows grouped by target host or chronological order.
    Guarantees 1-to-1 alignment with the input DataFrame rows so that len(X_seq) == len(df).

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned and normalized flow records.
    feature_cols : list
        List of numeric/encoded feature column names.
    target_col : str
        Label column to extract for sequence target (defaults to latest flow's label).
    host_col : str, optional
        Column name identifying target host (e.g., 'dst_host' or 'Destination IP').
    time_col : str, optional
        Timestamp column for sorting.
    window_size : int
        Number of consecutive flow steps in each sequence (default: 10).
    step_size : int
        Stride between sliding windows (default: 1).
    padding : str
        'pre' or 'post' zero-padding for bursts smaller than window_size.
        
    Returns:
    --------
    X_seq : np.ndarray
        Array of shape (len(df), window_size, len(feature_cols)).
    y_seq : np.ndarray
        Array of targets corresponding to each sequence window (len(df),).
    """
    N = len(df)
    F = len(feature_cols)
    if N == 0:
        return np.empty((0, window_size, F), dtype=np.float32), np.empty((0,), dtype=np.int64)

    feat_mat = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df[target_col].to_numpy() if target_col in df.columns else np.zeros(N, dtype=np.int64)

    X_seq = np.zeros((N, window_size, F), dtype=np.float32)
    y_seq = np.zeros(N, dtype=labels.dtype)

    if host_col and host_col in df.columns:
        host_vals = df[host_col].values
        groups = {}
        for idx, h in enumerate(host_vals):
            if h not in groups:
                groups[h] = []
            groups[h].append(idx)
        group_indices = list(groups.values())
    else:
        group_indices = [np.arange(N)]

    for indices in group_indices:
        indices = np.asarray(indices)
        M = len(indices)
        grp_feats = feat_mat[indices]
        grp_labels = labels[indices]
        
        for k in range(M):
            orig_idx = indices[k]
            y_seq[orig_idx] = grp_labels[k]
            if k < window_size - 1:
                pad_count = window_size - (k + 1)
                if padding == 'pre':
                    X_seq[orig_idx, pad_count:] = grp_feats[:k + 1]
                else:
                    X_seq[orig_idx, :k + 1] = grp_feats[:k + 1]
            else:
                X_seq[orig_idx] = grp_feats[k - window_size + 1 : k + 1]

    return X_seq, y_seq


def create_latent_sequence_dataset(latent_vectors, targets, window_size=10, step_size=1):
    """
    Construct sequential windows over Autoencoder latent embeddings.
    Allows recurrent sequence models (LSTM/GRU) to operate on compact manifold features.
    """
    num_samples = len(latent_vectors)
    sequences = []
    y_out = []
    
    for i in range(0, num_samples - window_size + 1, step_size):
        sequences.append(latent_vectors[i:i + window_size])
        y_out.append(targets[i + window_size - 1])
        
    return np.array(sequences, dtype=np.float32), np.array(y_out)
