"""
Data preprocessing, cleaning, encoding, and scaling pipeline for NSL-KDD dataset.
"""

import os
import pickle
import numpy as np
import pandas as pd
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
except ImportError:
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        def fit(self, X):
            arr = np.asarray(X, dtype=np.float32)
            self.mean_ = np.nanmean(arr, axis=0)
            self.scale_ = np.nanstd(arr, axis=0)
            self.scale_[self.scale_ == 0.0] = 1.0
            return self
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)
        def transform(self, X):
            arr = np.asarray(X, dtype=np.float32)
            mean = self.mean_ if self.mean_ is not None else np.nanmean(arr, axis=0)
            scale = self.scale_ if self.scale_ is not None else np.nanstd(arr, axis=0)
            scale[scale == 0.0] = 1.0
            return (arr - mean) / scale

    class LabelEncoder:
        def __init__(self):
            self.classes_ = []
        def fit(self, y):
            self.classes_ = np.unique(y)
            return self
        def transform(self, y):
            mapping = {c: i for i, c in enumerate(self.classes_)}
            return np.array([mapping.get(val, 0) for val in y])
        def fit_transform(self, y):
            self.fit(y)
            return self.transform(y)

COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

ATTACK_MAPPING = {
    'normal': 'Normal',
    # DoS
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']


def load_raw_data(train_path, test_path):
    """Load raw NSL-KDD training and testing files."""
    df_train = pd.read_csv(train_path, names=COLUMN_NAMES, header=None)
    df_test = pd.read_csv(test_path, names=COLUMN_NAMES, header=None)
    return df_train, df_test


def clean_data(df, is_train=True):
    """Clean dataframe: drop difficulty_level, remove duplicates from train, strip labels."""
    df = df.copy()
    if 'difficulty_level' in df.columns:
        df.drop(columns=['difficulty_level'], inplace=True)
    
    if is_train:
        df.drop_duplicates(inplace=True)
    
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    return df


def map_labels(df):
    """Map raw attack subtypes to 5 high-level categories and binary label."""
    df = df.copy()
    df['attack_category'] = df['label'].map(ATTACK_MAPPING).fillna('Unknown')
    df['binary_label'] = df['attack_category'].apply(lambda x: 0 if x == 'Normal' else 1)
    return df


def encode_features(train_df, test_df):
    """
    Perform one-hot encoding on categorical features with a unified vocabulary
    to guarantee identical columns across train and test sets.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    combined = pd.concat([train_df, test_df], axis=0, keys=['train', 'test'])
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLS, drop_first=False)
    
    train_encoded = combined.xs('train').copy()
    test_encoded = combined.xs('test').copy()
    
    return train_encoded, test_encoded


def scale_features(X_train, X_test, feature_cols):
    """Fit StandardScaler strictly on training features and transform train and test."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[feature_cols]),
                                  columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[feature_cols]),
                                 columns=feature_cols, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler


def run_full_preprocessing(raw_train_path, raw_test_path, output_dir, models_dir):
    """Complete end-to-end preprocessing pipeline with artifact serialization."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("1. Loading raw data...")
    train_raw, test_raw = load_raw_data(raw_train_path, raw_test_path)
    
    print("2. Cleaning data...")
    train_clean = clean_data(train_raw, is_train=True)
    test_clean = clean_data(test_raw, is_train=False)
    
    print("3. Mapping attack labels...")
    train_clean = map_labels(train_clean)
    test_clean = map_labels(test_clean)
    
    print("4. Encoding categorical features...")
    train_enc, test_enc = encode_features(train_clean, test_clean)
    
    non_feature_cols = ['label', 'attack_category', 'binary_label']
    feature_cols = [c for c in train_enc.columns if c not in non_feature_cols]
    
    print(f"Total features after one-hot encoding: {len(feature_cols)}")
    
    print("5. Scaling numeric features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(train_enc, test_enc, feature_cols)
    
    train_final = pd.concat([X_train_scaled, train_enc[non_feature_cols]], axis=1)
    test_final = pd.concat([X_test_scaled, test_enc[non_feature_cols]], axis=1)
    
    print("6. Fitting multiclass LabelEncoder...")
    le = LabelEncoder()
    le.fit(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])
    
    print("7. Serializing artifacts...")
    train_final.to_csv(f"{output_dir}/train_cleaned.csv", index=False)
    test_final.to_csv(f"{output_dir}/test_cleaned.csv", index=False)
    
    with open(f"{models_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{models_dir}/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    with open(f"{models_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
        
    print("Preprocessing completed successfully.")
    return train_final, test_final, scaler, feature_cols, le


# ==========================================
# 2. CIC-IDS2017 PREPROCESSING ADAPTER
# ==========================================
def clean_cicids2017(df):
    """
    Sanitize and prepare CIC-IDS2017 PCAP-extracted flow statistics.
    Handles column whitespace, division-by-zero Inf/NaN values, and binary/multiclass mapping.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    label_col = 'Label' if 'Label' in df.columns else [c for c in df.columns if 'label' in c.lower()][0]
    
    drop_cols = [c for c in ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Source Port'] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    raw_labels = df[label_col].astype(str).str.strip()
    df['attack_category'] = raw_labels
    df['binary_label'] = (raw_labels.str.lower() != 'benign').astype(int)
    
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['binary_label', label_col]]
    return df, feature_cols


# ==========================================
# 3. UNSW-NB15 PREPROCESSING ADAPTER
# ==========================================
UNSW_ATTACK_CATEGORIES = [
    'Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS',
    'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
]


def clean_unsw_nb15(df):
    """
    Clean and encode UNSW-NB15 network session records.
    Prepares binary label, attack category, and one-hot / frequency encodings for categorical fields.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
        
    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
        df['attack_cat'] = df['attack_cat'].replace({'': 'Normal', 'nan': 'Normal', 'NaN': 'Normal', 'Backdoor': 'Backdoors'})
        df['attack_category'] = df['attack_cat']
    else:
        df['attack_category'] = 'Unknown'
        
    if 'label' in df.columns:
        df['binary_label'] = df['label'].astype(int)
    elif 'attack_cat' in df.columns:
        df['binary_label'] = (df['attack_cat'].str.lower() != 'normal').astype(int)
    else:
        df['binary_label'] = 0
        
    cat_cols = [c for c in ['proto', 'service', 'state'] if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq).fillna(0.0)
        
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['binary_label', 'label']]
    return df, feature_cols
