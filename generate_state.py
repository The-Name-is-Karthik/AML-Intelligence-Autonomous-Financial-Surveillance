import pandas as pd
import joblib
from preprocess import load_data, engineer_features_stateless, perform_temporal_split, engineer_features_stateful

def main():
    DATA_PATH = "./SAML-D.csv"
    
    # 1. Load a portion of the data to calculate statistics
    # Note: For the most accurate prediction, this should be the same 
    # training set used to train your best_catboost_model.cbm.
    # If the file is 1GB, we'll load enough to get stable medians/means.
    print("Loading data to recreate preprocessing state...")
    df = load_data(DATA_PATH)
    
    # 2. Basic cleaning
    df = engineer_features_stateless(df)
    
    # 3. Split (using the same 0.2 split logic as training)
    train, test = perform_temporal_split(df, test_size=0.2)
    
    # 4. Generate State
    # This captures medians, account averages, and category encoders
    _, _, prep_state = engineer_features_stateful(train, test)
    
    # 5. Save State
    state_filename = "preprocessing_state.joblib"
    joblib.dump(prep_state, state_filename)
    
    print(f"\n✅ Success! {state_filename} has been created.")
    print("You can now build your Docker image.")

if __name__ == "__main__":
    main()
