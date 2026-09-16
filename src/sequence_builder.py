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
        Array of shape (num_sequences, window_size, len(feature_cols)).
    y_seq : np.ndarray
        Array of targets corresponding to each sequence window.
    """
    df_sorted = df.copy()
    
    # Sort chronologically if timestamp is present
    if time_col and time_col in df_sorted.columns:
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], errors='coerce')
        df_sorted.sort_values(by=time_col, inplace=True)
    
    sequences = []
    targets = []
    
    # Group by destination host if specified, otherwise slide globally
    if host_col and host_col in df_sorted.columns:
        groups = [group for _, group in df_sorted.groupby(host_col, sort=False)]
    else:
        groups = [df_sorted]
        
    for grp in groups:
        features = grp[feature_cols].values
        labels = grp[target_col].values if target_col in grp.columns else np.zeros(len(grp))
        num_rows = len(grp)
        
        if num_rows < window_size:
            # Pad short bursts
            pad_len = window_size - num_rows
            pad_shape = (pad_len, len(feature_cols))
            zeros_feat = np.zeros(pad_shape, dtype=np.float32)
            if padding == 'pre':
                seq = np.vstack([zeros_feat, features])
            else:
                seq = np.vstack([features, zeros_feat])
            sequences.append(seq)
            targets.append(labels[-1])
        else:
            for start_idx in range(0, num_rows - window_size + 1, step_size):
                end_idx = start_idx + window_size
                sequences.append(features[start_idx:end_idx])
                # Label is typically determined by the terminal event in the sequence window
                targets.append(labels[end_idx - 1])
                
    if len(sequences) == 0:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
    return np.array(sequences, dtype=np.float32), np.array(targets)


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
