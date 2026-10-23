import os
import pandas as pd
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    student_input: Dict[str, Any]
    estimated_rank: int
    filtered_colleges: List[Dict[str, Any]]
    ai_advice: str

def estimate_kcet_rank_node(state: AgentState) -> AgentState:
    inp = state["student_input"]
    pu_pct = inp["pu_science_marks"]
    kcet_pct = (inp["kcet_marks"] / 180.0) * 100.0
    composite_cet_score = (0.5 * pu_pct) + (0.5 * kcet_pct)

    if composite_cet_score >= 90:
        est_rank = int(1000 + (100 - composite_cet_score) * 400)
    elif composite_cet_score >= 75:
        est_rank = int(5000 + (90 - composite_cet_score) * 1800)
    elif composite_cet_score >= 60:
        est_rank = int(32000 + (75 - composite_cet_score) * 2800)
    elif composite_cet_score >= 45:
        est_rank = int(74000 + (60 - composite_cet_score) * 3500)
    else:
        est_rank = 140000

    state["estimated_rank"] = max(1, est_rank)
    return state

def filter_colleges_node(state: AgentState) -> AgentState:
    df = pd.read_csv("data/processed_colleges.csv")
    budget = state["student_input"]["per_year_invest"]
    rank = state["estimated_rank"]
    matched_records = []

    for _, row in df.iterrows():
        if rank <= row["cutoff_rank"] and row["govt_fee"] <= budget:
            matched_records.append({
                "college_name": row["college_name"],
                "district": row["district"],
                "gov_seat": int(row["dept_govt_seat"]),
                "private_seat": int(row["dept_pr_seat"]),
                "departments": f"{row['department']} (Total Intake: {row['total_intake']})",
                "seat_type_recommended": "KCET Govt Quota",
                "fee_pay": float(row["govt_fee"]),
                "cutoff_rank": int(row["cutoff_rank"]),
                "composite_rank_score": float(row["composite_rank_score"])
            })
        elif row["mgmt_fee"] <= budget:
            matched_records.append({
                "college_name": row["college_name"],
                "district": row["district"],
                "gov_seat": int(row["dept_govt_seat"]),
                "private_seat": int(row["dept_pr_seat"]),
                "departments": f"{row['department']} (Total Intake: {row['total_intake']})",
                "seat_type_recommended": "Private / Management Quota",
                "fee_pay": float(row["mgmt_fee"]),
                "cutoff_rank": int(row["cutoff_rank"]),
                "composite_rank_score": float(row["composite_rank_score"])
            })

    matched_records = sorted(matched_records, key=lambda x: x["composite_rank_score"], reverse=True)
    state["filtered_colleges"] = matched_records[:8]
    return state

def groq_advisory_node(state: AgentState) -> AgentState:
    inp = state["student_input"]
    rank = state["estimated_rank"]
    colleges = state["filtered_colleges"]

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        state["ai_advice"] = f"GROQ_API_KEY not configured. Estimated rank is {rank}. Focus on top-ranked Govt Quota choices."
        return state

    # Using Groq's ultra-fast Llama 3.3 model
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=groq_api_key)
    
    prompt = ChatPromptTemplate.from_template("""
    You are the Lead AI Education Counselor at 'Grow with Gyan', helping PUC Science students
    from rural Karnataka (Bidar, Kalaburagi, etc.) avoid financial misallocation.

    Student Profile:
    - Name: {name}
    - 10th Marks: {tenth}% | PUC Science Marks: {pu}% | KCET Score: {kcet}/180
    - Estimated KCET Rank: ~{rank}
    - Annual Budget Willing to Invest: ₹{budget}/year

    Top Matched Colleges & Branches (Sorted by Quality & ROI Score):
    {colleges}

    Provide a clear, empathetic, data-backed advisory covering:
    1. **Govt Seat vs Management Seat Trade-off**: Which top colleges they can secure via KCET Govt Quota vs where they can smartly use their budget.
    2. **ROI Comparison**: Warn them against spending ₹1.5L-2L on management seats in lower-ranked local colleges if they qualify for better options.
    3. **Actionable Next Steps for KEA Option Entry**.
    """)

    chain = prompt | llm
    response = chain.invoke({
        "name": inp["name"],
        "tenth": inp["tenth_marks"],
        "pu": inp["pu_science_marks"],
        "kcet": inp["kcet_marks"],
        "rank": rank,
        "budget": inp["per_year_invest"],
        "colleges": colleges
    })
    state["ai_advice"] = response.content
    return state

def build_gyan_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("estimate_rank", estimate_kcet_rank_node)
    workflow.add_node("filter_colleges", filter_colleges_node)
    workflow.add_node("groq_advisor", groq_advisory_node)

    workflow.set_entry_point("estimate_rank")
    workflow.add_edge("estimate_rank", "filter_colleges")
    workflow.add_edge("filter_colleges", "groq_advisor")
    workflow.add_edge("groq_advisor", END)

    return workflow.compile()

gyan_agent = build_gyan_graph()
