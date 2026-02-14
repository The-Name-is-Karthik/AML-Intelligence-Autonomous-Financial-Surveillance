import pandas as pd
import joblib
import os
import json
import time
from catboost import CatBoostClassifier
from preprocess import load_data, engineer_features_stateless, perform_temporal_split, engineer_features_stateful, prepare_for_model
from train_models import train_and_eval

# Constants
FEEDBACK_FILE = "human_feedback.csv"
MODEL_PATH = "best_catboost_model.cbm"
STATE_PATH = "preprocessing_state.joblib"
SCORE_FILE = "model_metadata.json"

def get_current_best_score():
    """Reads the current best PR-AUC from metadata."""
    if os.path.exists(SCORE_FILE):
        with open(SCORE_FILE, "r") as f:
            return json.load(f).get("best_pr_auc", 0.0)
    return 0.0

def save_new_best(model, prep_state, score):
    """Saves the new model, state, and updates the score metadata."""
    model.save_model(MODEL_PATH)
    joblib.dump(prep_state, STATE_PATH)
    with open(SCORE_FILE, "w") as f:
        json.dump({"best_pr_auc": score, "last_updated": str(pd.Timestamp.now())}, f)
    print(f"New best model saved with PR-AUC: {score:.4f}")

def run_retraining_cycle():
    """Runs one training cycle and promotes the model if it's better."""
    print(f"\n--- Checking for Retraining Cycle: {pd.Timestamp.now()} ---")
    
    # 1. Check if we have enough human feedback to justify retraining
    if not os.path.exists(FEEDBACK_FILE):
        print("Waiting for human feedback data...")
        return
    
    feedback_count = len(pd.read_csv(FEEDBACK_FILE))
    if feedback_count < 5: # Threshold: Retrain every 5 new feedback points
        print(f"Not enough new feedback ({feedback_count}/5). Skipping...")
        return

    print("Executing retraining with active learning feedback...")
    
    current_best = get_current_best_score()
    
    # 2. Data Loading & Augmentation
    df = load_data("SAML-D.csv")
    feedback_df = pd.read_csv(FEEDBACK_FILE)
    
    # Apply human feedback as ground truth
    for _, row in feedback_df.iterrows():
        df.loc[df['Sender_account'] == row['account_id'], 'Is_laundering'] = row['is_laundering']

    # 3. Pipeline
    df = engineer_features_stateless(df)
    train, test = perform_temporal_split(df, test_size=0.2)
    train, test, prep_state = engineer_features_stateful(train, test)
    
    train = prepare_for_model(train)
    test = prepare_for_model(test)
    
    cols_to_exclude = ['Is_laundering', 'Sender_account', 'Receiver_account', 'Laundering_type']
    X_train = train.drop(columns=[c for c in cols_to_exclude if c in train.columns])
    y_train = train['Is_laundering']
    X_test = test.drop(columns=[c for c in cols_to_exclude if c in test.columns])
    y_test = test['Is_laundering']

    # 4. Train
    new_model, new_score = train_and_eval("CatBoost_Challenger", X_train, y_train, X_test, y_test)
    
    # 5. Champion vs Challenger Comparison (A/B Testing)
    if new_score > current_best:
        save_new_best(new_model, prep_state, new_score)
        # Archive feedback after successful integration
        os.rename(FEEDBACK_FILE, f"archived_feedback_{int(time.time())}.csv")
        print("Champion promoted. Watchdog updated.")
    else:
        print(f" Challenger ({new_score:.4f}) did not beat Champion ({current_best:.4f}). Keeping current model.")

if __name__ == "__main__":
    # This loop runs the check every hour
    # For a demo, you can reduce this to 60 seconds
    CHECK_INTERVAL = 3600 # 1 hour
    
    while True:
        try:
            run_retraining_cycle()
        except Exception as e:
            print(f"Error in retraining cycle: {e}")
        
        print(f"Sleeping for {CHECK_INTERVAL / 60:.0f} minutes...")
        time.sleep(CHECK_INTERVAL)
