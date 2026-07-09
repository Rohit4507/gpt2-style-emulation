"""FastAPI application initialization and router inclusion."""
from fastapi import FastAPI

from app.api.routes import router
from app.core.logging import setup_logging
from app.core.middleware import setup_middleware

setup_logging()

app = FastAPI(
    title="Style Emulation API",
    description="Fine-tuned GPT-2 style-emulation generation and StyleSim scoring.",
    version="1.0.0",
)
setup_middleware(app)
app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "Style Emulation API is running. See /docs for the OpenAPI schema."}
