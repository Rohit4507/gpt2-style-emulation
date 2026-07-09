"""
Global middleware: CORS, rate limiting, and a catch-all exception handler.

Caveat (documented, not hidden): the rate limiter uses slowapi's default
in-memory backend. That's correct for a single-container Space, but it
resets on restart and does NOT coordinate across multiple replicas/workers.
If this ever runs with >1 worker or behind a load balancer, swap the
storage_uri below for a Redis-backed one (slowapi supports this natively).
"""
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings

logger = logging.getLogger(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"],
)


def setup_middleware(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
