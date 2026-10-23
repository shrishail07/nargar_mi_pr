from pydantic import BaseModel, Field
from typing import List

class StudentInput(BaseModel):
    name: str = Field(..., example="Ramesh Patil")
    tenth_marks: float = Field(..., ge=0, le=100, description="10th Percentage")
    pu_science_marks: float = Field(..., ge=0, le=100, description="PUC II PCM Percentage")
    kcet_marks: int = Field(..., ge=0, le=180, description="KCET Marks out of 180")
    per_year_invest: float = Field(..., ge=50000, description="Annual Budget in INR (e.g., 200000)")

class CollegeRecommendation(BaseModel):
    college_name: str
    district: str
    gov_seat: int
    private_seat: int
    departments: str
    seat_type_recommended: str
    fee_pay: float
    cutoff_rank: int
    composite_rank_score: float

class AdvisoryResponse(BaseModel):
    student_name: str
    estimated_kcet_rank: int
    recommendations: List[CollegeRecommendation]
    ai_counselor_advice: str
