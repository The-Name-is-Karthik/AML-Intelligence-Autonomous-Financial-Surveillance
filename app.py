import streamlit as st
import requests
import pandas as pd
import time
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AML Intelligence",
    layout="wide",
)

# ---STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0E1117;
    }
    
    .stMetric {
        background-color: #1E232E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363D;
    }
    
    .status-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF4B4B;
        background-color: #1E232E;
        margin-bottom: 1rem;
    }
    
    .status-clear {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28A745;
        background-color: #1E232E;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- API HELPERS ---
API_BASE = "http://localhost:8000"  

def get_prediction(txn_data):
    try:
        response = requests.post(f"{API_BASE}/predict", json=txn_data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_investigation(account_id, risk_score):
    try:
        response = requests.post(f"{API_BASE}/investigate?account_id={account_id}&risk_score={risk_score}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def send_feedback(account_id, is_laundering, risk_score):
    try:
        response = requests.post(
            f"{API_BASE}/feedback?account_id={account_id}&is_laundering={is_laundering}&risk_score={risk_score}"
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2833/2833314.png", width=80)
    st.title("AML Command Center")
    st.markdown("---")
    st.info("Phase 1-4 Operational")
    
    if st.button(" Reset Session"):
        st.session_state.clear()
        st.rerun()

# --- MAIN DASHBOARD ---
st.title(" AML Intelligence ")
st.subheader("Autonomous Financial Surveillance System")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Transaction Input")
    with st.expander(" Manual Transaction Entry", expanded=True):
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=7500.0)
        sender = st.number_input("Sender Account ID", value=1491989064)
        receiver = st.number_input("Receiver Account ID", value=99887766)
        payment_type = st.selectbox("Payment Type", ["Cross-border", "Transfer", "Cheque", "Debit card", "Credit card"])
        sender_loc = st.text_input("Sender Bank Location", value="UK")
        receiver_loc = st.text_input("Receiver Bank Location", value="Cayman Islands")
        
        test_txn = {
            "Time": "14:30:00",
            "Date": "2026-02-13",
            "Sender_account": sender,
            "Receiver_account": receiver,
            "Amount": amount,
            "Payment_currency": "USD",
            "Received_currency": "USD",
            "Sender_bank_location": sender_loc,
            "Receiver_bank_location": receiver_loc,
            "Payment_type": payment_type
        }
        
    if st.button(" Process Transaction", use_container_width=True):
        with st.spinner("Analyzing with Watchdog Model..."):
            result = get_prediction(test_txn)
            st.session_state.last_prediction = result
            st.session_state.last_txn = test_txn

with col2:
    if 'last_prediction' in st.session_state:
        pred = st.session_state.last_prediction
        score = pred.get('laundering_probability', 0)
        is_risky = pred.get('is_high_risk', False)
        
        st.markdown(f"### Phase 1: Watchdog Results")
        
        # Risk Meter
        st.progress(score)
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Laundering Probability", f"{score*100:.1f}%")
        status = "ALERT" if is_risky else "SECURE"
        m_col2.metric("System Status", status)
        
        # --- NEW: RISK DRIVERS ---
        st.markdown("#### Key Risk Drivers")
        drivers = pred.get('risk_drivers', [])
        if drivers:
            # Create a simple vertical bar chart for drivers
            driver_df = pd.DataFrame(drivers, columns=['Feature', 'Impact Score'])
            st.bar_chart(driver_df.set_index('Feature'), horizontal=True, height=200)
            st.caption("Positive scores indicate features that pushed the model toward predicting risk.")
        
        if is_risky:
            st.markdown(f'<div class="status-alert"> <b>HIGH RISK DETECTED</b><br>Score: {score:.4f}. Immediate investigation required.</div>', unsafe_allow_html=True)
            
            if st.button(" Trigger Agentic Investigation (Phase 2)", type="primary"):
                with st.spinner("Agent searching ledger and web..."):
                    inv_result = run_investigation(st.session_state.last_txn['Sender_account'], score)
                    st.session_state.inv_result = inv_result
        else:
            st.markdown(f'<div class="status-clear"><b>TRANSACTION CLEARED</b><br>Statistical behavior within normal parameters.</div>', unsafe_allow_html=True)

# --- PHASE 2 & 3: INVESTIGATION & REPORTING ---
if 'inv_result' in st.session_state:
    st.markdown("---")
    st.title(" Phase 2 & 3: Forensic Deep-Dive")
    
    inv = st.session_state.inv_result
    
    i_col1, i_col2 = st.columns([2, 1])
    
    with i_col1:
        st.markdown("### Agent Reasoning Log")
        st.success("Analysis Complete by Llama 3.3-70B")
        st.markdown(f"> {inv.get('investigation_summary', 'No summary available.')}")
    
    with i_col2:
        st.markdown("### Generated Report")
        report_path = inv.get('report_path', '')
        if report_path:
            st.info(f"Report Generated: {os.path.basename(report_path)}")
            # For a local streamlit app, we can't easily serve the file directly without a custom route,
            # but we can show the path.
            st.code(report_path)
            
        st.markdown("### Phase 4: Human Feedback")
        st.markdown("Does this investigation align with the actual risk?")
        
        f_col1, f_col2 = st.columns(2)
        if f_col1.button(" Confirm Crime", use_container_width=True):
            res = send_feedback(st.session_state.last_txn['Sender_account'], True, score)
            st.toast("Feedback Logged: True Negative prevented.")
            
        if f_col2.button(" False Positive", use_container_width=True):
            res = send_feedback(st.session_state.last_txn['Sender_account'], False, score)
            st.toast("Feedback Logged: Model will be desensitized.")

st.markdown("---")
st.caption("AML Intelligence Hub v2.0 | High-Throughput GBDT Model | Agentic LangGraph Pipeline")
