"""
FastAPI endpoints. ModelManager is injected via Depends (rather than
imported as a bare module-level singleton) specifically so tests can
override it with a lightweight fake and avoid downloading real
checkpoints -- see tests/conftest.py.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.schemas import (
    DriftRequest,
    DriftResponse,
    EvaluateRequest,
    EvaluateResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from app.core.config import settings
from app.core.middleware import limiter
from app.services import evaluator, generator
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()


def get_model_manager() -> ModelManager:
    return ModelManager()


@router.post("/generate", response_model=GenerateResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
def generate_endpoint(
    request: Request,
    body: GenerateRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> GenerateResponse:
    try:
        model = manager.get(body.topic, body.size)
    except Exception as exc:  # e.g. checkpoint missing from the Hub repo
        logger.exception("Failed to load checkpoint for %s/%s", body.topic, body.size)
        raise HTTPException(status_code=404, detail=f"Checkpoint unavailable: {exc}") from exc

    texts = [generator.generate(model, manager.tokenizer, body.prompt) for _ in range(body.n_samples)]
    return GenerateResponse(topic=body.topic, size=body.size, texts=texts)


@router.post("/evaluate", response_model=EvaluateResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
def evaluate_endpoint(request: Request, body: EvaluateRequest) -> EvaluateResponse:
    references = evaluator.get_references()
    if body.topic not in references:
        raise HTTPException(status_code=400, detail=f"No reference set for topic '{body.topic}'")

    score = evaluator.style_similarity(references[body.topic], body.texts)
    return EvaluateResponse(topic=body.topic, style_sim=score)


@router.post("/drift", response_model=DriftResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
def drift_endpoint(
    request: Request,
    body: DriftRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> DriftResponse:
    try:
        model = manager.get(body.topic, body.size)
    except Exception as exc:
        logger.exception("Failed to load checkpoint for %s/%s", body.topic, body.size)
        raise HTTPException(status_code=404, detail=f"Checkpoint unavailable: {exc}") from exc

    references = evaluator.get_references()
    if body.topic not in references:
        raise HTTPException(status_code=400, detail=f"No reference set for topic '{body.topic}'")
    ref_texts = references[body.topic]

    text = generator.generate(model, manager.tokenizer, prompt="")
    scores = [evaluator.style_similarity(ref_texts, [text])]
    for _ in range(body.n_hops - 1):
        text = generator.generate(model, manager.tokenizer, prompt=text)
        scores.append(evaluator.style_similarity(ref_texts, [text]))

    return DriftResponse(topic=body.topic, size=body.size, scores=scores)


@router.get("/health", response_model=HealthResponse)
def health(manager: ModelManager = Depends(get_model_manager)) -> HealthResponse:
    """Liveness/readiness probe for Docker/Spaces -- reports which
    checkpoints are currently resident in memory, not just that the
    process is running."""
    return HealthResponse(status="ok", resident_models=list(manager.resident_keys()))
