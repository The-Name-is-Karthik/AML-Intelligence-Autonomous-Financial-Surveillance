import pandas as pd
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from privacy_utils import deidentify_data
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Ledger
DATA_PATH = "SAML-D.csv"
df = pd.DataFrame()

if os.path.exists(DATA_PATH):
    # Load a subset for tool lookup or use unique index if possible
    # For now, we load a reasonably sized chunk for demo purposes
    df = pd.read_csv(DATA_PATH, nrows=50000)

@tool
def get_account_history(account_id: int):
    """
    Accesses the bank's internal ledger for a specific account.
    Returns a behavioral summary and the most recent transactions for this account.
    Useful for distinguishing between 'High Net Worth' individuals and 'Mules'.
    """
    if df.empty:
        return "Internal Error: Database not loaded."
    
    # Filter for both sender and receiver roles
    all_account_activity = df[(df['Sender_account'] == account_id) | (df['Receiver_account'] == account_id)]
    
    if all_account_activity.empty:
        return f"No transaction history found for account {account_id}."
    
    # 1. Behavioral Analytics (Profiling)
    avg_txn = all_account_activity['Amount'].mean()
    max_txn = all_account_activity['Amount'].max()
    total_txns = len(all_account_activity)
    incoming_count = len(all_account_activity[all_account_activity['Receiver_account'] == account_id])
    outgoing_count = len(all_account_activity[all_account_activity['Sender_account'] == account_id])
    
    summary = (
        f"--- Account Profile Summary ---\n"
        f"Total Transactions: {total_txns}\n"
        f"Average Transaction Size: ${avg_txn:,.2f}\n"
        f"Largest Transaction Found: ${max_txn:,.2f}\n"
        f"Incoming vs Outgoing Ratio: {incoming_count}/{outgoing_count}\n"
        f"-------------------------------\n"
    )
    
    # 2. Recent Raw History
    recent_history = all_account_activity.tail(10).to_string()
    
    # De-identify before returning to LLM
    full_report = summary + "\nRecent Transactions:\n" + recent_history
    return deidentify_data(full_report)

# Initialize External Search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
search_tool = TavilySearchResults(k=3) if TAVILY_API_KEY else None
