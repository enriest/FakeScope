import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


OPENAI_MODEL = os.getenv("FAKESCOPE_OPENAI_MODEL", "gpt-4o-mini")


def _build_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _truncate(s: str, max_chars: int = 3000) -> str:
    s = s or ""
    return s[:max_chars]


def generate_explanation(
    input_text: str,
    model_scores: Dict[str, float],
    google_items: List[Dict[str, Any]],
    google_score: Optional[float],
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    """
    Use OpenAI to generate a concise explanation combining model result and fact-check evidence.
    Returns empty string if OPENAI_API_KEY not set.
    """
    client = _build_client()
    if client is None:
        return ""  # OpenAI not configured

    evidence_lines = []
    for it in google_items[:5]:
        rating = it.get("textual_rating")
        url = it.get("url")
        pub = it.get("publisher")
        evidence_lines.append(f"- {rating} — {pub} — {url}")

    evidence_str = "\n".join(evidence_lines) if evidence_lines else "(no external fact checks found)"

    user_prompt = (
        "You are a careful fact-checking teacher.\n"
        "Explain in 2-4 short paragraphs, accessible to non-experts, why the claim/article might be true or fake.\n"
        "Use external evidence if available. Keep balanced, avoid overclaiming, and cite sources as bullet links.\n\n"
        f"Claim/Article text (truncated):\n{_truncate(input_text)}\n\n"
        f"Model scores (probabilities): fake={model_scores.get('fake'):.3f}, true={model_scores.get('true'):.3f}.\n"
        f"Google Fact Check aggregate score (0-1): {google_score if google_score is not None else 'N/A'}.\n\n"
        f"External evidence:\n{evidence_str}"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": "You are an expert, neutral fact-checking assistant. Be precise and cite sources.",
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()
