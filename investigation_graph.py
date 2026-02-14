import operator
from typing import Annotated, List, TypedDict, Union

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

from tools import get_account_history, search_tool

# State Definition
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    account_id: int
    risk_score: float

# Nodes
def call_detective(state: AgentState):
    # llama-3.3-70b-versatile is the current stable 70b model on Groq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    tools = [get_account_history]
    if search_tool:
        tools.append(search_tool)
    
    llm_with_tools = llm.bind_tools(tools)
    
    # Balanced AML Expert System Prompt
    system_msg = {
        "role": "system",
        "content": (
            "You are a Senior Digital Forensic AML Investigator at a global tier-1 bank. "
            "Your objective is to perform a balanced, multi-dimensional investigation on transaction alerts. "
            "You must remain OBJECTIVE: your goal is not to find crime, but to find the TRUTH."
            "\n\nINVESTIGATION FRAMEWORK:"
            "\n1. VOLUMETRIC ANALYSIS: Look for 'Structuring' ($9,000 range) vs 'Consistent Volume' (Are these amounts normal for this account?)."
            "\n2. VELOCITY ANALYSIS: Check for 'Rapid Movement' vs 'Business Cycles' (e.g., Is this just monthly payroll or a holiday fan-out?)."
            "\n3. NETWORK TOPOLOGY: Identify 'Flow-through' vs 'Legitimate Aggregation' (e.g., A merchant collecting small payments is normal)."
            "\n4. EXONERATION LOGIC: Actively seek reasons why this behavior might be LEGITIMATE. Check if the 'Account Profile Summary' shows a long history of this same pattern."
            "\n\nOPERATIONAL RULES:"
            "\n- First call 'get_account_history' to establish a 30-day baseline."
            "\n- Compare the current alert to the 'Average Transaction Size' in the profile summary."
            "\n- If the behavior (like a Fan-Out) is consistent with the account's historical average, you should lean toward a 'Low' or 'Medium' risk verdict unless OSINT shows negative news."
            "\n- Your final report must include: 'Risk Verdict', 'Rationale', and 'Exonerating Factors' (if any)."
        )
    }
    
    response = llm_with_tools.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "action"
    return END

# Graph Construction
workflow = StateGraph(AgentState)
workflow.add_node("detective", call_detective)

agent_tools = [get_account_history]
if search_tool:
    agent_tools.append(search_tool)
workflow.add_node("action", ToolNode(agent_tools))

workflow.set_entry_point("detective")
workflow.add_conditional_edges("detective", should_continue)
workflow.add_edge("action", "detective")

aml_app = workflow.compile()
