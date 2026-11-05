from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    score: int = Field(ge=0, le=10, description="Score from 0 to 10, where 10 is perfect")
    reason: str = Field(description="Brief explanation for the score")
    confidence: float = Field(ge=0.0, le=1.0, description="How confident the judge is, from 0.0 to 1.0")