#!/usr/bin/env python3
"""
List available Gemini models and whether they support generateContent.
Usage:
  source .venv/bin/activate
  source .env  # ensure GEMINI_API_KEY is set
  python scripts/list_gemini_models.py
"""
import os
import sys
import requests

def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Export it or add to .env.")
        sys.exit(1)
    url = "https://generativelanguage.googleapis.com/v1/models"
    try:
        r = requests.get(url, params={"key": api_key}, timeout=15)
        r.raise_for_status()
        models = (r.json() or {}).get("models", [])
    except Exception as e:
        print(f"Failed to list models: {e}")
        sys.exit(2)

    print(f"Found {len(models)} models")
    for m in models:
        name = m.get("name", "")  # e.g., models/gemini-1.5-flash
        supported = ",".join(m.get("supportedGenerationMethods", []) or [])
        input_tokens = (m.get("inputTokenLimit") or m.get("inputTokenLimit"))
        print(f"- {name} | methods=[{supported}] | input_tokens={input_tokens}")

    print("\nTip: Set FAKESCOPE_GEMINI_MODEL to one of the names above without the 'models/' prefix.")

if __name__ == "__main__":
    main()
