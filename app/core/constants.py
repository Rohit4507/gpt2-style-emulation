"""
Domain constants for the style-emulation project. Kept separate from
config.py because these describe the *model artifacts* (fixed by however
train_and_push.py was run), not environment-specific deployment settings.
"""
from typing import Dict, List

TOPICS: Dict[str, int] = {
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Sci/Tech": 3,
}

DATA_SIZES: List[int] = [25, 50, 100]


def slug(topic: str, size: int) -> str:
    """Must match the naming used by train_and_push.py when it uploaded
    checkpoints to the Hub (<topic>_<size> subfolders)."""
    return f"{topic.lower().replace('/', '')}_{size}"
