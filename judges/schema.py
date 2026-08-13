from pydantic import BaseModel, Field

class JudgeVerdict(BaseModel):
    score: int = Field(description="Score from 0 to 10, where 10 is perfect")
    reason: str = Field(description="Brief explanation for the score")
    confidence: float = Field(description="How confident the judge is, from 0.0 to 1.0")