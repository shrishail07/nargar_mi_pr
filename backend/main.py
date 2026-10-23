import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import StudentInput, AdvisoryResponse
from backend.agent import gyan_agent
from scripts.feature_engineering import run_feature_engineering

app = FastAPI(
    title="Grow with Gyan - AI College Advisory API",
    version="1.0.0",
    description="FastAPI + LangGraph backend for mapping Karnataka PUC students to optimal VTU colleges."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    if not os.path.exists("data/processed_colleges.csv"):
        run_feature_engineering()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Grow with Gyan API"}

@app.post("/api/v1/recommend", response_model=AdvisoryResponse)
async def recommend_colleges(student: StudentInput):
    try:
        initial_state = {
            "student_input": student.model_dump(),
            "estimated_rank": 0,
            "filtered_colleges": [],
            "ai_advice": ""
        }
        result = gyan_agent.invoke(initial_state)

        return AdvisoryResponse(
            student_name=student.name,
            estimated_kcet_rank=result["estimated_rank"],
            recommendations=result["filtered_colleges"],
            ai_counselor_advice=result["ai_advice"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
