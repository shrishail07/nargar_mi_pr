import os
import streamlit as st
import pandas as pd
import requests

API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1/recommend")

st.set_page_config(
    page_title="Grow with Gyan | KCET & VTU AI Advisor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Grow with Gyan: Smart VTU & KCET College Advisor")
st.markdown(
    "**Empowering PUC Science Graduates across Karnataka with Transparent ROI, Cutoff & Fee Insights**"
)

with st.sidebar:
    st.header("📋 Student Academic & Budget Profile")
    with st.form("student_input_form"):
        name = st.text_input("Student Full Name", value="Basavaraj Patil")
        tenth_marks = st.number_input("10th Std Marks (%)", min_value=35.0, max_value=100.0, value=84.5, step=0.5)
        pu_science_marks = st.number_input("PUC II Science (PCM %)", min_value=35.0, max_value=100.0, value=82.0, step=0.5)
        kcet_marks = st.number_input("KCET Marks (Out of 180)", min_value=1, max_value=180, value=88, step=1)
        per_year_invest = st.number_input(
            "Max Annual Fee Investment (₹ / Year)",
            min_value=50000,
            max_value=600000,
            value=200000,
            step=10000
        )
        submitted = st.form_submit_button("🔍 Find Best ROI Colleges")

if submitted:
    with st.spinner("Running LangGraph Agent & Gemini 1.5 Pro ROI Analysis..."):
        payload = {
            "name": name,
            "tenth_marks": tenth_marks,
            "pu_science_marks": pu_science_marks,
            "kcet_marks": kcet_marks,
            "per_year_invest": per_year_invest
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Top Metrics Bar
            col1, col2, col3 = st.columns(3)
            col1.metric("Student Name", data["student_name"])
            col2.metric("Estimated KCET Rank", f"#{data['estimated_kcet_rank']:,}")
            col3.metric("Annual Budget Cap", f"₹{per_year_invest:,.0f}/yr")

            st.subheader("🏛️ Optimal College & Seat Allocation Matrix")
            recs = data["recommendations"]
            if recs:
                df_res = pd.DataFrame(recs)
                # Select & format exact columns requested by customer
                display_df = df_res[[
                    "college_name",
                    "district",
                    "gov_seat",
                    "private_seat",
                    "departments",
                    "seat_type_recommended",
                    "fee_pay",
                    "composite_rank_score"
                ]].rename(columns={
                    "college_name": "College Name",
                    "district": "District",
                    "gov_seat": "Govt Seats (Dept)",
                    "private_seat": "Private Seats (Dept)",
                    "departments": "Department & Intake",
                    "seat_type_recommended": "Recommended Quota",
                    "fee_pay": "Annual Fee to Pay (₹)",
                    "composite_rank_score": "Gyan Quality Score (10)"
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No matching colleges found within this budget and rank combination.")

            st.subheader("🤖 Gemini 1.5 Pro Personalized Counseling Report")
            st.info(data["ai_counselor_advice"])

        except Exception as e:
            st.error(f"Error connecting to FastAPI Backend: {e}")
else:
    st.info("👈 Fill in your marks and annual budget in the sidebar to generate your customized VTU College Roadmap.")
