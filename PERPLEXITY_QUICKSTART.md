# Quick Reference: Perplexity Integration

## Setup (30 seconds)

```bash
# 1. Get API key from https://www.perplexity.ai/settings/api
export PERPLEXITY_API_KEY="pplx-your-key-here"

# 2. Select provider
export FAKESCOPE_LLM_PROVIDER="perplexity"

# 3. Run (must activate venv first!)
source .venv/bin/activate
python -m streamlit run src/app.py
```

## Prompt Location

**File**: `src/openai_explain.py`  
**Lines**: 83-92

```python
# Line 83: System prompt (AI role)
system_prompt = "You are an expert, neutral fact-checking assistant..."

# Lines 85-92: User prompt (task + data)
user_prompt = (
    "You are a careful fact-checking teacher.\n"
    "Explain in 2-4 short paragraphs..."
    f"Claim/Article text: {_truncate(input_text)}\n\n"
    f"Model scores: fake={model_scores.get('fake'):.3f}..."
)
```

## Quick Changes

### Make it shorter
```python
user_prompt = f"In 1 sentence, is this fake? {_truncate(input_text, 200)}"
```

### Add confidence score
```python
user_prompt = f"Rate 0-100 and explain:\n{_truncate(input_text)}"
```

### Change model
```bash
export FAKESCOPE_PERPLEXITY_MODEL="llama-3.1-sonar-small-128k-online"  # Faster
```

## Switch Providers

```bash
# Use OpenAI
export FAKESCOPE_LLM_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."

# Use Perplexity
export FAKESCOPE_LLM_PROVIDER="perplexity"
export PERPLEXITY_API_KEY="pplx-..."

# Use Google Gemini
export FAKESCOPE_LLM_PROVIDER="gemini"
export GEMINI_API_KEY="your-gemini-key"
```

## Full Docs

- **Prompt examples**: `PROMPT_CUSTOMIZATION.md`
- **Deployment**: `GUIDE.md`
- **Summary**: `LLM_INTEGRATION_SUMMARY.md`
