import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Memory efficient load with specified dtypes."""
    print("Loading dataset...")
    dtypes = {
        'Sender_account': 'uint64',
        'Receiver_account': 'uint64',
        'Amount': 'float32',
        'Is_laundering': 'int8',
        'Payment_type': 'category',
        'Sender_bank_location': 'category',
        'Receiver_bank_location': 'category',
        'Payment_currency': 'category',
        'Received_currency': 'category'
    }
    
    # Load once
    df = pd.read_csv(file_path, dtype=dtypes)
    
    # 2. TEMPORAL PREP
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    # Combine for sorting
    df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    
    # Basic cleaning
    categorical_cols = ['Payment_type', 'Sender_bank_location', 'Receiver_bank_location']
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("Unknown")
        
    return df.sort_values('Timestamp').reset_index(drop=True)

def engineer_features_stateless(df):
    """Features that do NOT depend on dataset statistics (safe to do before split)."""
    print("Engineering stateless features...")
    
    # Time features
    df['hour_of_day'] = pd.to_datetime(df['Timestamp']).dt.hour
    df['day_of_week_enc'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
    
    # Domain Logic
    df['currency_match'] = (df['Payment_currency'].astype(str) == df['Received_currency'].astype(str)).astype(int)
    df['is_cross_border'] = (df['Sender_bank_location'].astype(str) != df['Receiver_bank_location'].astype(str)).astype(int)
    
    # AML Domain: Round amounts (laundering often uses nice round numbers)
    df['is_round_10'] = (df['Amount'] % 10 == 0).astype(int)
    df['is_round_100'] = (df['Amount'] % 100 == 0).astype(int)
    
    # High risk destinations
    high_risk = ["Mexico", "UAE", "Turkey", "Cayman Islands"]
    df['is_high_risk_dest'] = df['Receiver_bank_location'].isin(high_risk).astype(int)
    
    return df

def perform_temporal_split(df, test_size=0.2):
    """Split based on time to simulate real-world 'future' prediction."""
    print(f"Performing temporal split (test_size={test_size})...")
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

class PreprocessingState:
    """Stores training statistics to ensure inference matches training."""
    def __init__(self):
        self.amt_median = None
        self.outlier_bounds = None # (lower, upper)
        self.sender_avg_mapping = None
        self.label_encoders = {}
        self.known_classes = {}

def engineer_features_stateful(train_df, test_df):
    """Features that depend on statistics. Fit on Train, Apply to Test."""
    print("Engineering stateful features (Leakage Prevention)...")
    state = PreprocessingState()
    
    # 1. Median Imputation
    state.amt_median = train_df['Amount'].median()
    train_df['Amount'] = train_df['Amount'].fillna(state.amt_median)
    test_df['Amount'] = test_df['Amount'].fillna(state.amt_median)
    
    # 2. Outlier Detection
    Q1 = train_df['Amount'].quantile(0.25)
    Q3 = train_df['Amount'].quantile(0.75)
    IQR = Q3 - Q1
    state.outlier_bounds = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    lower, upper = state.outlier_bounds
    
    train_df['is_amount_outlier'] = ((train_df['Amount'] < lower) | (train_df['Amount'] > upper)).astype(int)
    test_df['is_amount_outlier'] = ((test_df['Amount'] < lower) | (test_df['Amount'] > upper)).astype(int)
    
    # 3. Account Behavior
    state.sender_avg_mapping = train_df.groupby('Sender_account')['Amount'].mean().to_dict()
    
    train_df['avg_amt_sender'] = train_df['Sender_account'].map(state.sender_avg_mapping)
    test_df['avg_amt_sender'] = test_df['Sender_account'].map(state.sender_avg_mapping)
    
    test_df['avg_amt_sender'] = test_df['avg_amt_sender'].fillna(state.amt_median)
    
    train_df['amt_relative_to_avg'] = train_df['Amount'] / (train_df['avg_amt_sender'] + 1e-6)
    test_df['amt_relative_to_avg'] = test_df['Amount'] / (test_df['avg_amt_sender'] + 1e-6)
    
    # 4. Encoding
    cat_to_encode = ['Payment_type', 'Sender_bank_location', 'Receiver_bank_location']
    for col in cat_to_encode:
        le = LabelEncoder()
        train_df[f'{col}_enc'] = le.fit_transform(train_df[col].astype(str))
        state.label_encoders[col] = le
        state.known_classes[col] = set(le.classes_)
        
        test_df[f'{col}_enc'] = test_df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in state.known_classes[col] else -1
        )
        
    return train_df, test_df, state

def preprocess_single_row(row_dict, state: PreprocessingState):
    """Transforms a single raw transaction dictionary into model-ready features."""
    df = pd.DataFrame([row_dict])
    
    # 1. Temporal/Stateless
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    df['Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    
    df['hour_of_day'] = df['Timestamp'].dt.hour
    df['day_of_week_enc'] = df['Timestamp'].dt.dayofweek
    
    df['currency_match'] = (df['Payment_currency'] == df['Received_currency']).astype(int)
    df['is_cross_border'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)
    df['is_round_10'] = (df['Amount'] % 10 == 0).astype(int)
    df['is_round_100'] = (df['Amount'] % 100 == 0).astype(int)
    
    high_risk = ["Mexico", "UAE", "Turkey", "Cayman Islands"]
    df['is_high_risk_dest'] = df['Receiver_bank_location'].isin(high_risk).astype(int)
    
    # 2. Stateful (Using saved state)
    df['Amount'] = df['Amount'].fillna(state.amt_median)
    lower, upper = state.outlier_bounds
    df['is_amount_outlier'] = ((df['Amount'] < lower) | (df['Amount'] > upper)).astype(int)
    
    sender = row_dict.get('Sender_account')
    avg_amt = state.sender_avg_mapping.get(sender, state.amt_median)
    df['avg_amt_sender'] = avg_amt
    df['amt_relative_to_avg'] = df['Amount'] / (avg_amt + 1e-6)
    
    # Label Encoding
    for col, le in state.label_encoders.items():
        val = str(row_dict.get(col, "Unknown"))
        df[f'{col}_enc'] = le.transform([val])[0] if val in state.known_classes[col] else -1
        
    # 3. Final Cleanup
    cols_to_keep = ['Amount', 'hour_of_day', 'day_of_week_enc', 'currency_match', 
                    'is_cross_border', 'is_round_10', 'is_round_100', 'is_high_risk_dest', 
                    'is_amount_outlier', 'avg_amt_sender', 'amt_relative_to_avg', 
                    'Payment_type_enc', 'Sender_bank_location_enc', 'Receiver_bank_location_enc']
    
    return df[cols_to_keep]

def prepare_for_model(df):
    """Final column selection and cleanup. Keeps Laundering_type for analysis but mark for removal in X."""
    drop_cols = ['Time', 'Date', 'Timestamp', 'Payment_type', 
                 'Sender_bank_location', 'Receiver_bank_location',
                 'Payment_currency', 'Received_currency']
    return df.drop(columns=[c for c in drop_cols if c in df.columns])

if __name__ == "__main__":
    # This shows the intended pipeline flow
    DATA_PATH = "./SAML-D.csv" # Replace with actual path
    
    # full_df = load_data(DATA_PATH)
    # full_df = engineer_features_stateless(full_df)
    
    # train, test = perform_temporal_split(full_df)
    # train, test = engineer_features_stateful(train, test)
    
    # train_final = prepare_for_model(train)
    # test_final = prepare_for_model(test)
    
    # print(f"Train shape: {train_final.shape}, Test shape: {test_final.shape}")