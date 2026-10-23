import os
import glob
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)

def get_youtube_review_score(college_name: str) -> float:
    """
    Searches YouTube for college reviews, extracts transcripts, and uses Gemini
    to generate a normalized quality score (1.0 to 10.0) for ranking.
    """
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    if not youtube_api_key:
        return 7.0  # Fallback baseline score if API key is not configured

    try:
        youtube = build("youtube", "v3", developerKey=youtube_api_key)
        req = youtube.search().list(
            q=f"{college_name} Karnataka engineering review placements KCET",
            part="snippet",
            type="video",
            maxResults=2
        )
        res = req.execute()
        transcripts = []
        for item in res.get("items", []):
            vid_id = item["id"]["videoId"]
            try:
                t_list = YouTubeTranscriptApi.get_transcript(vid_id, languages=['en', 'kn', 'hi'])
                transcripts.append(" ".join([t["text"] for t in t_list[:150]]))
            except Exception:
                continue

        if not transcripts:
            return 6.5

        prompt = PromptTemplate.from_template(
            "Analyze the following YouTube review transcripts for {college}. "
            "Rate the college from 1.0 to 10.0 based on real-world ROI, placements, faculty, "
            "and absence of hidden fees. Return ONLY the numeric float.\n\nTranscripts: {text}"
        )
        chain = prompt | llm
        score_str = chain.invoke({"college": college_name, "text": " ".join(transcripts)}).content.strip()
        return float(score_str)
    except Exception:
        return 6.5

def generate_sample_district_data():
    """Generates sample Excel structure matching your exact specification if no files exist."""
    os.makedirs("data/raw", exist_ok=True)
    sample_data = [
        {
            "district": "Kalaburagi",
            "college_name": "PDA College of Engineering, Kalaburagi",
            "total_intake": 660,
            "total_govt_seat": 420,
            "total_private_seat": 240,
            "CS_govt_seat": 90, "CS_pr_seat": 30, "CS_cutoff": 22000, "CS_govt_fee": 96500, "CS_mgmt_fee": 220000,
            "EC_govt_seat": 90, "EC_pr_seat": 30, "EC_cutoff": 35000, "EC_govt_fee": 96500, "EC_mgmt_fee": 160000,
            "ME_govt_seat": 60, "ME_pr_seat": 30, "ME_cutoff": 85000, "ME_govt_fee": 96500, "ME_mgmt_fee": 110000,
        },
        {
            "district": "Kalaburagi",
            "college_name": "Appa Institute of Engineering and Technology, Kalaburagi",
            "total_intake": 360,
            "total_govt_seat": 210,
            "total_private_seat": 150,
            "CS_govt_seat": 60, "CS_pr_seat": 30, "CS_cutoff": 48000, "CS_govt_fee": 96500, "CS_mgmt_fee": 150000,
            "EC_govt_seat": 60, "EC_pr_seat": 30, "EC_cutoff": 65000, "EC_govt_fee": 96500, "EC_mgmt_fee": 125000,
            "ME_govt_seat": 30, "ME_pr_seat": 15, "ME_cutoff": 120000, "ME_govt_fee": 96500, "ME_mgmt_fee": 95000,
        },
        {
            "district": "Bidar",
            "college_name": "Guru Nanak Dev Engineering College, Bidar",
            "total_intake": 480,
            "total_govt_seat": 300,
            "total_private_seat": 180,
            "CS_govt_seat": 75, "CS_pr_seat": 25, "CS_cutoff": 42000, "CS_govt_fee": 96500, "CS_mgmt_fee": 165000,
            "EC_govt_seat": 60, "EC_pr_seat": 30, "EC_cutoff": 58000, "EC_govt_fee": 96500, "EC_mgmt_fee": 135000,
            "ME_govt_seat": 45, "ME_pr_seat": 15, "ME_cutoff": 110000, "ME_govt_fee": 96500, "ME_mgmt_fee": 100000,
        },
        {
            "district": "Bengaluru",
            "college_name": "Nitte Meenakshi Institute of Technology (NMIT), Bengaluru",
            "total_intake": 900,
            "total_govt_seat": 540,
            "total_private_seat": 360,
            "CS_govt_seat": 120, "CS_pr_seat": 60, "CS_cutoff": 9500, "CS_govt_fee": 104000, "CS_mgmt_fee": 450000,
            "EC_govt_seat": 90, "EC_pr_seat": 30, "EC_cutoff": 16500, "EC_govt_fee": 104000, "EC_mgmt_fee": 280000,
            "ME_govt_seat": 60, "ME_pr_seat": 30, "ME_cutoff": 45000, "ME_govt_fee": 104000, "ME_mgmt_fee": 175000,
        },
        {
            "district": "Tumakuru",
            "college_name": "Siddaganga Institute of Technology (SIT), Tumakuru",
            "total_intake": 840,
            "total_govt_seat": 520,
            "total_private_seat": 320,
            "CS_govt_seat": 120, "CS_pr_seat": 40, "CS_cutoff": 7800, "CS_govt_fee": 104000, "CS_mgmt_fee": 400000,
            "EC_govt_seat": 90, "EC_pr_seat": 30, "EC_cutoff": 13500, "EC_govt_fee": 104000, "EC_mgmt_fee": 240000,
            "ME_govt_seat": 60, "ME_pr_seat": 30, "ME_cutoff": 38000, "ME_govt_fee": 104000, "ME_mgmt_fee": 160000,
        }
    ]
    df = pd.DataFrame(sample_data)
    df.to_excel("data/raw/karnataka_colleges.xlsx", index=False)

def run_feature_engineering():
    excel_files = glob.glob("data/raw/*.xlsx")
    if not excel_files:
        generate_sample_district_data()
        excel_files = glob.glob("data/raw/*.xlsx")

    df_list = [pd.read_excel(f) for f in excel_files]
    raw_df = pd.concat(df_list, ignore_index=True)

    departments = ["CS", "EC", "ME"]
    normalized_rows = []

    for _, row in raw_df.iterrows():
        yt_score = get_youtube_review_score(row["college_name"])
        for dept in departments:
            if f"{dept}_govt_seat" in row:
                cutoff = row[f"{dept}_cutoff"]
                # Feature 6: Composite Rank Score (70% KEA Cutoff Weight + 30% YouTube Review Score)
                cutoff_norm = max(0, (150000 - cutoff) / 150000) * 10
                composite_score = round((0.7 * cutoff_norm) + (0.3 * yt_score), 2)

                normalized_rows.append({
                    "district": row["district"],
                    "college_name": row["college_name"],
                    "total_intake": row["total_intake"],
                    "total_govt_seat": row["total_govt_seat"],
                    "total_private_seat": row["total_private_seat"],
                    "department": dept,
                    "dept_govt_seat": int(row[f"{dept}_govt_seat"]),
                    "dept_pr_seat": int(row[f"{dept}_pr_seat"]),
                    "cutoff_rank": int(cutoff),
                    "govt_fee": float(row[f"{dept}_govt_fee"]),
                    "mgmt_fee": float(row[f"{dept}_mgmt_fee"]),
                    "yt_sentiment_score": yt_score,
                    "composite_rank_score": composite_score
                })

    processed_df = pd.DataFrame(normalized_rows)
    processed_df.sort_values(by="composite_rank_score", ascending=False, inplace=True)
    processed_df.to_csv("data/processed_colleges.csv", index=False)
    print("Feature engineering completed. Saved to data/processed_colleges.csv")

if __name__ == "__main__":
    run_feature_engineering()
