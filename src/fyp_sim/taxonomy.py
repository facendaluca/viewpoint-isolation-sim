from __future__ import annotations

import re
from typing import Final


def normalise_token(token: str) -> str:
    """
    Normalise a free-form token into a stable key:
    - lower
    - replace non-alphanumeric with underscores
    - collapse repeats
    - strip edges

    Examples:
        "Health & Wellness" -> "health_wellness"
        "DIY & Life hacks" -> "diy_life_hacks"
    """
    s = token.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


# Canonical topic keys derived from survey Q3 options
TOPIC_CATEGORIES: Final[tuple[str, ...]] = (
    "politics",
    "international_news",
    "social_issues_activism",
    "health_wellness",
    "finance_economics",
    "comedy_memes",
    "diy_life_hacks",
    "fashion_beauty",
    "sports",
)

# Human-readable labels for plots/tables later
TOPIC_LABELS: Final[dict[str, str]] = {
    "politics": "Politics",
    "international_news": "International News",
    "social_issues_activism": "Social Issues & Activism",
    "health_wellness": "Health & Wellness",
    "finance_economics": "Finance & Economics",
    "comedy_memes": "Comedy & Memes",
    "diy_life_hacks": "DIY & Life Hacks",
    "fashion_beauty": "Fashion & Beauty",
    "sports": "Sports",
}
