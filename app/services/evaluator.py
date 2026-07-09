"""
StyleSim (mean pairwise cosine, Eq. 2) scoring, plus the held-out
reference bundles pulled from the same Hub repo the checkpoints live in.

The embedder and references are module-level, lazily-initialized
singletons -- there's only one of each needed regardless of how many
GPT-2 checkpoints are resident, so they don't go through ModelManager's
LRU logic.
"""
import json
import logging
import threading
from typing import Dict, List

from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

logger = logging.getLogger(__name__)

_embedder: SentenceTransformer = None
_embedder_lock = threading.Lock()

_references: Dict[str, List[str]] = None
_references_lock = threading.Lock()


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                logger.info("Loading MiniLM embedder...")
                _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_references() -> Dict[str, List[str]]:
    global _references
    if _references is None:
        with _references_lock:
            if _references is None:
                logger.info("Downloading reference_texts.json from %s", settings.model_repo)
                path = hf_hub_download(
                    settings.model_repo,
                    filename="reference_texts.json",
                    token=settings.hf_token,
                )
                with open(path) as f:
                    _references = json.load(f)
    return _references


def style_similarity(references: List[str], generations: List[str]) -> float:
    """Mean pairwise cosine between generated and reference embeddings."""
    embedder = get_embedder()
    ref_emb = embedder.encode(references)
    gen_emb = embedder.encode(generations)
    return float(cosine_similarity(gen_emb, ref_emb).mean())
