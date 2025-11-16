import os
import time
from typing import Any, Dict, List, Optional

import requests


FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")
FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Normalization map (0.0-1.0). Extend as needed based on observed ratings.
_RATING_MAP = {
    "true": 1.0,
    "mostly true": 0.85,
    "mostly-true": 0.85,
    "half true": 0.6,
    "half-true": 0.6,
    "mixture": 0.5,
    "mixed": 0.5,
    "unclear": 0.5,
    "misleading": 0.4,
    "mostly false": 0.25,
    "mostly-false": 0.25,
    "false": 0.0,
    "pants on fire": 0.0,
    "pants-fire": 0.0,
}


def _normalize_rating(textual_rating: Optional[str]) -> Optional[float]:
    if not textual_rating:
        return None
    key = textual_rating.strip().lower()
    return _RATING_MAP.get(key)


def is_configured() -> bool:
    """Return True if the Google Fact Check API key is available."""
    return bool(FACTCHECK_API_KEY)


def fetch_fact_checks(
    claim_text: str,
    language_code: str = "en",
    max_results: int = 5,
    timeout: float = 8.0,
    retries: int = 2,
) -> List[Dict[str, Any]]:
    """
    Query Google Fact Check Tools API and return normalized items.

    Returns list of dicts: {claim_text, textual_rating, rating_score, title, url, publisher, review_date}
    """
    if not FACTCHECK_API_KEY:
        return []

    params = {
        "query": claim_text,
        "languageCode": language_code,
        "pageSize": max_results,
        "key": FACTCHECK_API_KEY,
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(FACTCHECK_ENDPOINT, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                items = []
                for c in data.get("claims", [])[:max_results]:
                    reviews = c.get("claimReview", []) or []
                    # Take the best/first review
                    if not reviews:
                        continue
                    r = reviews[0]
                    textual_rating = (r.get("textualRating") or "").strip()
                    items.append(
                        {
                            "claim_text": c.get("text"),
                            "textual_rating": textual_rating,
                            "rating_score": _normalize_rating(textual_rating),
                            "title": r.get("title"),
                            "url": r.get("url"),
                            "publisher": (r.get("publisher", {}) or {}).get("name"),
                            "review_date": r.get("reviewDate"),
                        }
                    )
                return items
            else:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.5 * (attempt + 1))

    # On failure, return empty array rather than raising to keep app responsive
    return []


def aggregate_google_score(items: List[Dict[str, Any]]) -> Optional[float]:
    """Aggregate rating_score to a single 0-1 value (mean of available ratings)."""
    scores = [it.get("rating_score") for it in items if isinstance(it.get("rating_score"), (int, float))]
    if not scores:
        return None
    return float(sum(scores) / len(scores))
