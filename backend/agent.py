import os
import pandas as pd
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Define the LangGraph State
class AgentState(TypedDict):
    student_input: Dict[str, Any]
    estimated_rank: int
    filtered_colleges: List[Dict[str, Any]]
    ai_advice: str

def estimate_kcet_rank_node(state: AgentState) -> AgentState:
    """
    Node 1: KEA calculates Engineering rank using 50% PU PCM marks + 50% KCET marks.
    Maps the composite score to an estimated KCET Rank.
    """
    inp = state["student_input"]
    pu_pct = inp["pu_science_marks"]
    kcet_pct = (inp["kcet_marks"] / 180.0) * 100.0
    composite_cet_score = (0.5 * pu_pct) + (0.5 * kcet_pct)

    # Realistic piecewise approximation of Karnataka CET Rank vs Aggregate Score
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
    """
    Node 2: Evaluates both Government Quota (Rank + Fee check) and
    Management Quota (Budget check) to prevent rural students from overpaying
    at low-tier colleges when a higher-ranked college fits their ₹2L budget.
    """
    df = pd.read_csv("data/processed_colleges.csv")
    budget = state["student_input"]["per_year_invest"]
    rank = state["estimated_rank"]

    matched_records = []

    for _, row in df.iterrows():
        # Case 1: Eligible for Government Quota Seat via KCET Rank AND within budget
        if rank <= row["cutoff_rank"] and row["govt_fee"] <= budget:
            matched_records.append({
                "college_name": row["college_name"],
                "district": row["district"],
                "gov_seat": int(row["dept_govt_seat"]),
                "private_seat": int(row["dept_pr_seat"]),
                "departments": f"{row['department']} (Total College Intake: {row['total_intake']})",
                "seat_type_recommended": "KCET Govt Quota",
                "fee_pay": float(row["govt_fee"]),
                "cutoff_rank": int(row["cutoff_rank"]),
                "composite_rank_score": float(row["composite_rank_score"])
            })
        # Case 2: Rank not sufficient for Govt seat, but Management seat fits Annual Budget!
        elif row["mgmt_fee"] <= budget:
            matched_records.append({
                "college_name": row["college_name"],
                "district": row["district"],
                "gov_seat": int(row["dept_govt_seat"]),
                "private_seat": int(row["dept_pr_seat"]),
                "departments": f"{row['department']} (Total College Intake: {row['total_intake']})",
                "seat_type_recommended": "Private / Management Quota",
                "fee_pay": float(row["mgmt_fee"]),
                "cutoff_rank": int(row["cutoff_rank"]),
                "composite_rank_score": float(row["composite_rank_score"])
            })

    # Sort by highest quality (composite_rank_score) first
    matched_records = sorted(matched_records, key=lambda x: x["composite_rank_score"], reverse=True)
    state["filtered_colleges"] = matched_records[:8]
    return state

def gemini_advisory_node(state: AgentState) -> AgentState:
    """
    Node 3: Gemini 1.5 Pro synthesizes a personalized ROI & Seat Allocation strategy.
    """
    inp = state["student_input"]
    rank = state["estimated_rank"]
    colleges = state["filtered_colleges"]

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        state["ai_advice"] = (
            "GOOGLE_API_KEY not found in environment. Based on deterministic filtering, "
            f"your estimated KCET rank is {rank}. Prioritize top-ranked Govt Quota options first."
        )
        return state

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("""
    You are the Lead AI Education Counselor at 'Grow with Gyan', helping PUC Science students
    from rural Karnataka (Bidar, Kalaburagi, etc.) avoid financial misallocation.

    Student Profile:
    - Name: {name}
    - 10th Marks: {tenth}% | PUC Science Marks: {pu}% | KCET Score: {kcet}/180
    - Estimated KCET Rank: ~{rank}
    - Annual Budget Willing to Invest: ₹{budget}/year

    Top Matched Colleges & Branches (Sorted by Quality & YouTube ROI Rank):
    {colleges}

    Provide a clear, empathetic, data-backed advisory covering:
    1. **Govt Seat vs Management Seat Trade-off**: Which top colleges they can secure via KCET Govt Quota (~₹96.5k-1.04L) vs where they can smartly use their ₹{budget} budget for a higher-ranked college.
    2. **ROI Comparison**: Warn them against spending ₹1.5L-2L on management seats in lower-ranked colleges if they qualify for a Govt seat in a better-ranked institution or a higher-tier branch in Bengaluru/Tumakuru/Kalaburagi.
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
    workflow.add_node("gemini_advisor", gemini_advisory_node)

    workflow.set_entry_point("estimate_rank")
    workflow.add_edge("estimate_rank", "filter_colleges")
    workflow.add_edge("filter_colleges", "gemini_advisor")
    workflow.add_edge("gemini_advisor", END)

    return workflow.compile()

gyan_agent = build_gyan_graph()
