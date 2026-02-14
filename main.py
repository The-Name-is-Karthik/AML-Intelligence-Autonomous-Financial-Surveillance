from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from preprocess import preprocess_single_row, PreprocessingState
from investigation_graph import aml_app
from report_utils import generate_sar_pdf
import os
import csv

app = FastAPI(title="AML Intelligence Hub")

FEEDBACK_FILE = "human_feedback.csv"

# 1. Load Model and State during startup
MODEL_PATH = "best_catboost_model.cbm"
STATE_PATH = "preprocessing_state.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(STATE_PATH):
    raise RuntimeError(f"Model or State file missing! Need {MODEL_PATH} and {STATE_PATH}")

watchdog = CatBoostClassifier().load_model(MODEL_PATH)
prep_state = joblib.load(STATE_PATH)

# 2. Define Request Schema
class Transaction(BaseModel):
    Time: str
    Date: str
    Sender_account: int
    Receiver_account: int
    Amount: float
    Payment_currency: str
    Received_currency: str
    Sender_bank_location: str
    Receiver_bank_location: str
    Payment_type: str

@app.post("/predict")
async def predict(txn: Transaction):
    try:
        txn_dict = txn.dict()
        processed_features = preprocess_single_row(txn_dict, prep_state)
        probability = float(watchdog.predict_proba(processed_features)[0][1])
        
        # Calculate Risk Drivers (SHAP values)
        feature_names = processed_features.columns.tolist()
        
        # Use a Pool to ensure data types are handled correctly by CatBoost
        predict_pool = Pool(data=processed_features, cat_features=watchdog.get_cat_feature_indices())
        
        shap_values = watchdog.get_feature_importance(
            data=predict_pool, 
            type='ShapValues'
        )
        
        # CatBoost returns (n_samples, n_features + 1) for ShapValues
        if len(shap_values.shape) > 1:
            importance_row = shap_values[0][:-1]
        else:
            importance_row = shap_values[:-1]
            
        # Create a dictionary of {feature: importance_score}
        drivers = dict(zip(feature_names, importance_row))
        # Sort and take top 3 positive contributors to risk
        sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "laundering_probability": probability,
            "is_high_risk": probability > 0.85,
            "risk_drivers": sorted_drivers
        }
    except Exception as e:
        print(f"--- PREDICT ERROR ---")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/investigate")
async def investigate(account_id: int, risk_score: float = 0.0):
    """
    Triggers the Agentic Deep-Dive for a specific account.
    Phase 2: The Detective Agent
    """
    try:
        inputs = {
            "messages": [("user", f"Investigate account {account_id}. Risk score is {risk_score}.")],
            "account_id": account_id,
            "risk_score": risk_score
        }
        
        # We run the graph synchronously for the API response
        result = aml_app.invoke(inputs)
        
        # Extract the final response from the agent
        final_message = result["messages"][-1].content
        
        # Phase 3: Generate the formal SAR Report
        report_path = generate_sar_pdf(account_id, risk_score, final_message)
        
        return {
            "account_id": account_id,
            "investigation_summary": final_message,
            "report_path": report_path,
            "status": "completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def log_feedback(account_id: int, is_laundering: bool, risk_score: float):
    """
    Logs human feedback to improve the model in Phase 4.
    """
    try:
        file_exists = os.path.isfile(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['account_id', 'is_laundering', 'risk_score', 'timestamp'])
            writer.writerow([account_id, int(is_laundering), risk_score, pd.Timestamp.now()])
        
        return {"status": "Feedback logged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}