"""
Singleton model manager: loads GPT-2 checkpoints from the Hub on demand
and keeps at most `settings.max_resident_models` of them in memory at
once (LRU eviction). Twelve GPT-2-small checkpoints at ~500MB each would
otherwise accumulate to ~6GB resident once every topic/size combo has
been requested -- fine on a beefy server, not fine on a free CPU Space.
"""
import logging
import os
import threading
from collections import OrderedDict
from typing import Tuple

from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from app.core.config import settings
from app.core.constants import slug

logger = logging.getLogger(__name__)


class ModelManager:
    """Process-wide singleton. Thread-safe get() with LRU eviction."""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        logger.info("Initializing ModelManager (max_resident=%d)", settings.max_resident_models)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._models: "OrderedDict[str, GPT2LMHeadModel]" = OrderedDict()
        self._models_lock = threading.Lock()

    def get(self, topic: str, size: int) -> GPT2LMHeadModel:
        """Return the checkpoint for (topic, size), loading + caching it if needed."""
        key = slug(topic, size)
        with self._models_lock:
            if key in self._models:
                self._models.move_to_end(key)  # mark as most-recently-used
                return self._models[key]

        model = self._load_from_hub(key)

        with self._models_lock:
            self._models[key] = model
            self._models.move_to_end(key)
            self._evict_if_needed()
        return model

    def _load_from_hub(self, key: str) -> GPT2LMHeadModel:
        logger.info("Loading checkpoint from Hub: %s", key)
        config_path = hf_hub_download(
            settings.model_repo,
            filename="config.json",
            subfolder=key,
            token=settings.hf_token,
        )
        folder = os.path.dirname(config_path)
        model = GPT2LMHeadModel.from_pretrained(folder)
        model.eval()
        return model

    def _evict_if_needed(self) -> None:
        while len(self._models) > settings.max_resident_models:
            evicted_key, _ = self._models.popitem(last=False)  # oldest = least recently used
            logger.info("Evicting checkpoint from memory: %s", evicted_key)

    def resident_keys(self) -> Tuple[str, ...]:
        with self._models_lock:
            return tuple(self._models.keys())
