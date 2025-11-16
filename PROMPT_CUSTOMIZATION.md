# LLM Prompt Customization Guide

## Quick Reference

### Where are the prompts?
**File**: `src/openai_explain.py`  
**Function**: `generate_explanation()` (lines ~60-90)

### Supported LLM Providers
- **OpenAI** (GPT-4o-mini, GPT-4, etc.)
- **Perplexity** (Llama 3.1 Sonar models)
- **Google Gemini** (Gemini 1.5 Flash, Pro, etc.)

All three use the same prompt structure.

---

## Prompt Locations

### 1. System Prompt (Line ~67)
```python
system_prompt = "You are an expert, neutral fact-checking assistant. Be precise and cite sources."
```

**What it does**: Defines the AI's role and behavior  
**Customization**: Change tone, expertise level, or special instructions

**Examples**:
```python
# Casual educator
system_prompt = "You're a friendly teacher helping people spot fake news."

# Technical expert
system_prompt = "You are a senior data scientist specializing in misinformation detection."

# Multilingual
system_prompt = "You are a multilingual fact-checker. Always respond in the input language."
```

---

### 2. User Prompt (Lines ~69-77)
```python
user_prompt = (
    "You are a careful fact-checking teacher.\n"
    "Explain in 2-4 short paragraphs, accessible to non-experts, why the claim/article might be true or fake.\n"
    "Use external evidence if available. Keep balanced, avoid overclaiming, and cite sources as bullet links.\n\n"
    f"Claim/Article text (truncated):\n{_truncate(input_text)}\n\n"
    f"Model scores (probabilities): fake={model_scores.get('fake'):.3f}, true={model_scores.get('true'):.3f}.\n"
    f"Google Fact Check aggregate score (0-1): {google_score if google_score is not None else 'N/A'}.\n\n"
    f"External evidence:\n{evidence_str}"
)
```

**What it does**: Contains the task description and all context data  
**Customization**: Change output format, length, focus areas, or structure

**Examples**:
```python
# Brief summary
user_prompt = f"Is this claim true or false? {_truncate(input_text, 500)}\n\nProvide a 2-sentence verdict."

# Detailed analysis
user_prompt = (
    "Analyze this claim in detail:\n\n"
    "1. CLAIM SUMMARY\n"
    "2. ML MODEL PREDICTION (with confidence)\n"
    "3. EXTERNAL FACT-CHECKER CONSENSUS\n"
    "4. SUPPORTING EVIDENCE\n"
    "5. CONTRADICTING EVIDENCE\n"
    "6. FINAL VERDICT (with reasoning)\n\n"
    f"Claim: {_truncate(input_text)}\n"
    f"Model: {model_scores}\n"
    f"Fact-checkers: {evidence_str}"
)

# JSON output
user_prompt = (
    "Return a JSON object with these keys:\n"
    '{"verdict": "true/false/mixed", "confidence": 0-100, "reasoning": "...", "sources": []}\n\n'
    f"Claim: {_truncate(input_text)}"
)
```

---

## Configuration Parameters

### Temperature (Line ~28, default: 0.2)
```python
temperature: float = 0.2
```

**Scale**: 0.0 (deterministic) to 2.0 (very creative)  
**Recommended for fact-checking**: 0.1 - 0.3

| Value | Behavior |
|-------|----------|
| 0.0-0.2 | Highly consistent, factual, focused |
| 0.3-0.5 | Balanced, slight variation |
| 0.6-1.0 | More creative, varied phrasing |
| 1.0-2.0 | Very creative (not recommended for facts) |

### Max Tokens (Line ~29, default: 500)
```python
max_tokens: int = 500
```

**Guide**:
- **100-200**: Very brief (1-2 sentences)
- **300-500**: Short explanation (2-4 paragraphs) ← Current default
- **800-1200**: Detailed analysis
- **1500-2000**: Comprehensive report

---

## Environment Variables

### Selecting LLM Provider
```bash
# Use OpenAI (default)
export FAKESCOPE_LLM_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."

# Use Perplexity
export FAKESCOPE_LLM_PROVIDER="perplexity"
export PERPLEXITY_API_KEY="pplx-..."
```

### Customizing Models
```bash
# OpenAI models
export FAKESCOPE_OPENAI_MODEL="gpt-4o-mini"        # Default, fast, cheap
export FAKESCOPE_OPENAI_MODEL="gpt-4o"             # More capable, slower, expensive
export FAKESCOPE_OPENAI_MODEL="gpt-3.5-turbo"     # Older, cheaper

# Perplexity models
export FAKESCOPE_PERPLEXITY_MODEL="llama-3.1-sonar-large-128k-online"   # Default, with search
export FAKESCOPE_PERPLEXITY_MODEL="llama-3.1-sonar-small-128k-online"   # Faster, cheaper
export FAKESCOPE_PERPLEXITY_MODEL="llama-3.1-70b-instruct"              # No search
```

---

## Testing Your Prompts

### Quick Test Script
Save as `test_prompt.py`:

```python
import os
os.environ["FAKESCOPE_LLM_PROVIDER"] = "openai"  # or "perplexity"
os.environ["OPENAI_API_KEY"] = "sk-..."  # your key

from src.openai_explain import generate_explanation

# Test case
explanation = generate_explanation(
    input_text="Breaking: Scientists discover cure for cancer",
    model_scores={"fake": 0.7, "true": 0.3},
    google_items=[
        {"textual_rating": "False", "publisher": "Snopes", "url": "https://snopes.com/..."}
    ],
    google_score=0.2,
    temperature=0.2,
    max_tokens=500
)

print(explanation)
```

Run:
```bash
python test_prompt.py
```

### Testing in Streamlit
```bash
source .venv/bin/activate
streamlit run src/app.py
```
1. Enter test claim
2. Click "Run Analysis"
3. Check "Explanation (LLM)" section

---

## Common Prompt Patterns

### 1. Structured Output
```python
user_prompt = (
    "Format your response as:\n"
    "VERDICT: [True/False/Mixed/Unverifiable]\n"
    "CONFIDENCE: [Low/Medium/High]\n"
    "KEY EVIDENCE: [bullet points]\n"
    "REASONING: [2-3 sentences]\n\n"
    f"Claim: {_truncate(input_text)}"
)
```

### 2. Simplified Explanation
```python
user_prompt = (
    "Explain in simple terms whether this is true or false:\n"
    f"{_truncate(input_text)}\n\n"
    "Use short sentences. Avoid jargon. Cite sources if available."
)
```

### 3. Evidence-Focused
```python
user_prompt = (
    "List the evidence for and against this claim:\n\n"
    f"CLAIM: {_truncate(input_text)}\n\n"
    f"SUPPORTING: [extract from evidence]\n"
    f"CONTRADICTING: [extract from evidence]\n"
    f"SOURCES: {evidence_str}"
)
```

### 4. Confidence Rating
```python
user_prompt = (
    "Rate this claim's credibility from 0-100 and explain why:\n"
    f"Claim: {_truncate(input_text)}\n\n"
    "Format:\n"
    "Score: [0-100]\n"
    "Reasoning: [2-3 sentences]\n"
    "Key factors: [bullet points]"
)
```

### 5. Comparative Analysis
```python
user_prompt = (
    "Compare our ML model's prediction with fact-checker consensus:\n\n"
    f"ML Model says: {model_scores.get('true'):.0%} likely TRUE\n"
    f"Fact-checkers say: {evidence_str}\n\n"
    "Where do they agree? Where do they differ? What's most likely true?"
)
```

---

## Provider-Specific Tips

### OpenAI (GPT-4o-mini)
- Very good at following structured instructions
- Fast responses (1-2 seconds)
- Use lower temperature (0.1-0.3) for consistency
- Best for: Concise, formatted output
- Cost: ~$0.01-0.03 per explanation

### Perplexity (Sonar models)
- Includes real-time web search
- May cite additional sources automatically
- Slightly slower (2-4 seconds)
- Best for: Current events, finding recent sources
- Note: "online" models do web search, others don't
- Cost: ~$0.01-0.05 per explanation

### Google Gemini (1.5 Flash)
- **Generous free tier**: 1500 requests/day
- Very fast responses (1-2 seconds)
- Good at natural, conversational output
- Best for: High volume, development, cost savings
- Multimodal capable (future: can process images)
- Cost: **FREE** up to 1500/day, then ~$0.005-0.01 per explanation
- Note: Safety filters may block certain content (rare in fact-checking)

---

## Deployment

After modifying prompts in `src/openai_explain.py`:

### Local
```bash
source .venv/bin/activate
python -m streamlit run src/app.py
```

### Docker
```bash
docker build -t fakescope:latest .
docker run --rm -p 8080:8080 \
  -e FAKESCOPE_LLM_PROVIDER="openai" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  fakescope:latest
```

### Fly.io
```bash
flyctl deploy
```

---

## Troubleshooting

### Empty responses
- Check API key is set: `echo $OPENAI_API_KEY`
- Verify provider: `echo $FAKESCOPE_LLM_PROVIDER`
- Check logs for errors

### Poor quality responses
- Increase `max_tokens` (try 800-1000)
- Adjust temperature (try 0.3-0.5)
- Add more specific instructions to prompt
- Try different model (GPT-4o vs GPT-4o-mini)

### Inconsistent formatting
- Lower temperature (try 0.1)
- Add explicit format instructions
- Use structured output pattern (see above)

### Rate limit errors
- OpenAI: Check quota at https://platform.openai.com/usage
- Perplexity: Check dashboard for limits
- Add caching to reduce API calls

---

## Best Practices

1. **Test iteratively**: Start simple, add complexity gradually
2. **Use examples**: Show the AI what you want in the prompt
3. **Be specific**: "Explain in 3 bullet points" > "Explain briefly"
4. **Version control**: Keep track of prompt changes
5. **A/B test**: Compare different prompts with same inputs
6. **Monitor costs**: Track API usage, especially with longer outputs

---

For more details, see:
- Full setup guide: `GUIDE.md`
- Project documentation: `Documents/fakescope-complete.md`
- Code implementation: `src/openai_explain.py`
