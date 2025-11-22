import os
import logging
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI
import json
import requests

# Load .env automatically if present so API keys are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# Environment variables for LLM configuration
OPENAI_MODEL = os.getenv("FAKESCOPE_OPENAI_MODEL", "gpt-4o-mini")
PERPLEXITY_MODEL = os.getenv("FAKESCOPE_PERPLEXITY_MODEL", "sonar-pro")
GEMINI_MODEL = os.getenv("FAKESCOPE_GEMINI_MODEL", "gemini-1.5-flash")
LLM_PROVIDER = os.getenv("FAKESCOPE_LLM_PROVIDER", "openai")  # Options: "openai", "perplexity", or "gemini"

# Logger setup (inherits root level; app config sets level)
logger = logging.getLogger(__name__)


def _build_openai_client() -> Optional[OpenAI]:
    """Build OpenAI client if API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _build_perplexity_client() -> Optional[OpenAI]:
    """Build Perplexity client using OpenAI SDK (Perplexity API is OpenAI-compatible)."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")


def _build_gemini_client():
    """Build Google Gemini client if API key is configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except ImportError:
        return None


def _gemini_list_models(api_key: str) -> List[Dict[str, Any]]:
    try:
        url = "https://generativelanguage.googleapis.com/v1/models"
        resp = requests.get(url, params={"key": api_key}, timeout=10)
        if resp.ok:
            data = resp.json() or {}
            return data.get("models", [])
    except Exception:
        pass
    return []


def _gemini_pick_candidates(preferred: Optional[str], api_key: Optional[str]) -> List[str]:
    # Reasonable fallback order covering common SKUs and aliases
    base_order = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
    ]
    # Start with preferred if provided
    candidates = []
    if preferred:
        candidates.append(preferred)
    for m in base_order:
        if m not in candidates:
            candidates.append(m)

    # If we can list models, filter to those that support generateContent
    if api_key:
        models = _gemini_list_models(api_key)
        supported = set()
        for m in models:
            name = m.get("name", "")  # e.g., models/gemini-1.5-flash
            methods = m.get("supportedGenerationMethods", []) or []
            if "generateContent" in methods and name.startswith("models/"):
                supported.add(name.split("/", 1)[1])
        if supported:
            candidates = [m for m in candidates if m in supported] + [m for m in supported if m not in candidates]
    return candidates


def _gemini_generate_via_rest(model: str, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        },
    }
    try:
        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=20)
        if not resp.ok:
            return None
        data = resp.json() or {}
        cands = data.get("candidates") or []
        if not cands:
            return None
        parts = (((cands[0] or {}).get("content") or {}).get("parts") or [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "\n".join([t for t in texts if t]).strip() or None
    except Exception:
        return None


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
    Use OpenAI, Perplexity, or Google Gemini API to generate a concise explanation combining model result and fact-check evidence.
    Returns empty string if no API key is configured.
    
    LLM provider is selected via FAKESCOPE_LLM_PROVIDER environment variable.
    Set to "openai" (default), "perplexity", or "gemini".
    """
    # Select provider based on CURRENT environment variable each call (avoids stale imports in Streamlit)
    provider = os.getenv("FAKESCOPE_LLM_PROVIDER", LLM_PROVIDER).lower()
    
    if provider == "gemini":
        client = _build_gemini_client()
        model = GEMINI_MODEL
        provider_name = "Gemini"
        use_gemini = True
    elif provider == "perplexity":
        client = _build_perplexity_client()
        model = PERPLEXITY_MODEL
        provider_name = "Perplexity"
        use_gemini = False
    else:
        client = _build_openai_client()
        model = OPENAI_MODEL
        provider_name = "OpenAI"
        use_gemini = False
    
    if client is None:
        return ""  # No API configured

    evidence_lines = []
    for it in google_items[:5]:
        rating = it.get("textual_rating")
        url = it.get("url")
        pub = it.get("publisher")
        evidence_lines.append(f"- {rating} — {pub} — {url}")

    evidence_str = "\n".join(evidence_lines) if evidence_lines else "(no external fact checks found)"

    # ============================================================
    # PROMPT LOCATION FOR OPENAI, PERPLEXITY, AND GEMINI:
    # This is where you can customize the prompts sent to the LLM.
    # The system_prompt sets the role/behavior of the AI.
    # The user_prompt contains the actual task and context.
    # ============================================================
    
    system_prompt = "You are an expert, neutral fact-checking assistant. Be precise and cite sources."
    
    user_prompt = (
        "You are a careful fact-checking teacher.\n"
        "Explain in 2-4 short paragraphs, accessible to non-experts, why the claim/article might be true or fake.\n"
        "Use external evidence if available. Keep balanced, avoid overclaiming, and cite sources as bullet links.\n\n"
        f"Claim/Article text (truncated):\n{_truncate(input_text)}\n\n"
        f"Model scores (probabilities): fake={model_scores.get('fake'):.3f}, true={model_scores.get('true'):.3f}.\n"
        f"Google Fact Check aggregate score (0-1): {google_score if google_score is not None else 'N/A'}.\n\n"
        f"External evidence:\n{evidence_str}"
    )

    try:
        if use_gemini:
            api_key = os.getenv("GEMINI_API_KEY")
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Try configured and fallback Gemini models via SDK first
            candidates = _gemini_pick_candidates(model, api_key)
            last_err = None
            for m in candidates:
                try:
                    gemini_model = client.GenerativeModel(
                        model_name=m,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                        },
                    )
                    response = gemini_model.generate_content(full_prompt)
                    text = getattr(response, "text", "") or ""
                    if text.strip():
                        logger.info(f"LLM explanation generated via {provider_name} model={m} chars={len(text.strip())}")
                        return text.strip()
                except Exception as ex:
                    last_err = ex
                    continue

            # As a final fallback, try REST API with the same candidates
            for m in candidates:
                text = _gemini_generate_via_rest(m, full_prompt, temperature, max_tokens)
                if text:
                    logger.info(f"LLM explanation generated via {provider_name} (REST) model={m} chars={len(text)}")
                    return text

            if last_err:
                raise last_err
            raise RuntimeError("Gemini generation failed for all candidate models")
        else:
            # OpenAI-compatible API (OpenAI and Perplexity)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content.strip()
            logger.info(f"LLM explanation generated via {provider_name} model={model} chars={len(content)}")
            return content
    except openai.RateLimitError:
        return f"(LLM explanation unavailable: {provider_name} quota exceeded.)"
    except Exception as e:
        return f"(LLM explanation unavailable due to {provider_name} API error: {str(e)})"
