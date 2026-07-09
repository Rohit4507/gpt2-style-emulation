"""Pydantic models for request/response validation, kept separate from
routes.py so they're reusable and independently testable."""
from typing import List, Literal

from pydantic import BaseModel, Field

Topic = Literal["World", "Sports", "Business", "Sci/Tech"]
TrainingSize = Literal[25, 50, 100]


class GenerateRequest(BaseModel):
    topic: Topic
    size: TrainingSize = 100
    n_samples: int = Field(5, ge=1, le=20)
    prompt: str = ""


class GenerateResponse(BaseModel):
    topic: Topic
    size: TrainingSize
    texts: List[str]


class EvaluateRequest(BaseModel):
    topic: Topic
    texts: List[str] = Field(..., min_length=1)


class EvaluateResponse(BaseModel):
    topic: Topic
    style_sim: float


class DriftRequest(BaseModel):
    topic: Topic
    size: TrainingSize = 100
    n_hops: int = Field(5, ge=2, le=10)


class DriftResponse(BaseModel):
    topic: Topic
    size: TrainingSize
    scores: List[float]


class HealthResponse(BaseModel):
    status: str
    resident_models: List[str]
