# Google Gemini API Integration Guide

## Why Use Gemini?

**Google Gemini** offers several advantages for FakeScope:
- **Generous Free Tier**: 1500 requests/day (vs. paid-only for OpenAI/Perplexity)
- **Fast & Efficient**: Gemini 1.5 Flash is optimized for speed
- **Multimodal**: Can process text, images, and video (future expansion possibilities)
- **Cost-Effective**: Even paid tier is 5x cheaper than GPT-4o-mini
- **Latest Tech**: Built on Google's DeepMind research

## Quick Setup (2 minutes)

### 1. Get API Key

1. Go to **https://ai.google.dev/**
2. Click **"Get API key in Google AI Studio"**
3. Sign in with your Google account
4. Click **"Create API Key"**
5. Copy the API key (starts with `AI...`)

### 2. Configure Environment

```bash
# Set Gemini as your LLM provider
export GEMINI_API_KEY="your-api-key-here"
export FAKESCOPE_LLM_PROVIDER="gemini"

# Optional: Choose model (default: gemini-1.5-flash)
export FAKESCOPE_GEMINI_MODEL="gemini-1.5-flash"
```

### 3. Install Dependencies

```bash
pip install google-generativeai
```

### 4. Run the App

```bash
source .venv/bin/activate
python -m streamlit run src/app.py
```

---

## Available Gemini Models

| Model | Best For | Speed | Cost (per 1M tokens) |
|-------|----------|-------|---------------------|
| **gemini-1.5-flash** | General use, fast responses | ⚡⚡⚡ | Input: $0.075, Output: $0.30 |
| gemini-1.5-pro | Complex reasoning | ⚡⚡ | Input: $1.25, Output: $5.00 |
| gemini-1.0-pro | Balanced performance | ⚡⚡ | Free tier available |

**Recommendation**: Use `gemini-1.5-flash` (default) for FakeScope—it's fast, cheap, and perfect for fact-checking explanations.

---

## Free Tier Limits

| Limit Type | Gemini 1.5 Flash | Gemini 1.5 Pro |
|------------|-----------------|----------------|
| Requests/minute | 15 | 2 |
| Requests/day | 1,500 | 50 |
| Tokens/minute | 1M | 32K |

**For FakeScope**: 1,500 requests/day is enough for ~150-200 users per day with moderate usage.

---

## Comparison: Gemini vs. OpenAI vs. Perplexity

| Feature | Gemini (Flash) | OpenAI (GPT-4o-mini) | Perplexity (Sonar) |
|---------|---------------|---------------------|-------------------|
| **Free tier** | ✅ 1500/day | ❌ Paid only | ❌ Paid only |
| **Speed** | ⚡⚡⚡ Very fast | ⚡⚡⚡ Very fast | ⚡⚡ Fast |
| **Cost/explanation** | ~$0.005-0.01 | ~$0.01-0.03 | ~$0.01-0.05 |
| **Real-time search** | ❌ No | ❌ No | ✅ Yes |
| **Multimodal** | ✅ Text, image, video | ✅ Text, image | ❌ Text only |
| **Context window** | 1M tokens | 128K tokens | 128K tokens |
| **Best for** | High volume, low cost | Production stability | Current events |

**Winner for FakeScope**: Gemini 1.5 Flash (best value, free tier, fast)

---

## Testing Your Setup

### Quick Test Script

```bash
cat > test_gemini.py << 'EOF'
import os
os.environ["FAKESCOPE_LLM_PROVIDER"] = "gemini"
os.environ["GEMINI_API_KEY"] = "your-key-here"

from src.openai_explain import generate_explanation

result = generate_explanation(
    input_text="Scientists discover breakthrough in renewable energy",
    model_scores={"fake": 0.2, "true": 0.8},
    google_items=[],
    google_score=None
)

print("Gemini Response:")
print(result)
EOF

python test_gemini.py
```

**Expected output**: A fact-checking explanation from Gemini.

---

## Advanced Configuration

### Adjusting Temperature & Tokens

In `src/openai_explain.py`, the `generate_explanation()` function accepts:

```python
temperature: float = 0.2,  # Lower = more focused (0.0-1.0 for Gemini)
max_tokens: int = 500,     # Max output length
```

**For Gemini specifically**:
- Temperature range: 0.0 to 2.0 (same as OpenAI)
- Recommended: 0.1-0.3 for fact-checking
- Max tokens: Up to 8192 (but 500 is sufficient)

### Using Different Models

```bash
# Fastest (default)
export FAKESCOPE_GEMINI_MODEL="gemini-1.5-flash"

# Best reasoning
export FAKESCOPE_GEMINI_MODEL="gemini-1.5-pro"

# Balanced (older)
export FAKESCOPE_GEMINI_MODEL="gemini-1.0-pro"
```

---

## Deployment with Gemini

### Local Docker

```bash
docker run --rm -p 8080:8080 \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -e FAKESCOPE_LLM_PROVIDER="gemini" \
  -e GOOGLE_FACTCHECK_API_KEY="$GOOGLE_FACTCHECK_API_KEY" \
  fakescope:latest
```

### Fly.io

```bash
# Set secrets
flyctl secrets set GEMINI_API_KEY="your-key"
flyctl secrets set FAKESCOPE_LLM_PROVIDER="gemini"

# Deploy
flyctl deploy
```

### Hugging Face Spaces

1. Go to Space → Settings → Repository secrets
2. Add:
   - **Name**: `GEMINI_API_KEY`, **Value**: `your-key`
   - **Name**: `FAKESCOPE_LLM_PROVIDER`, **Value**: `gemini`
3. Push code changes

---

## Troubleshooting

### "LLM explanation unavailable: Gemini API error"

**Cause**: API key not set or invalid

**Solution**:
```bash
# Check if key is set
echo $GEMINI_API_KEY

# Verify provider
echo $FAKESCOPE_LLM_PROVIDER

# Test API key manually
python -c "import google.generativeai as genai; genai.configure(api_key='your-key'); print(genai.list_models())"
```

### "ModuleNotFoundError: No module named 'google.generativeai'"

**Cause**: Library not installed

**Solution**:
```bash
pip install google-generativeai
```

### Rate Limit Errors

**Error**: "429 Resource exhausted"

**Cause**: Exceeded free tier limits (15 req/min or 1500/day)

**Solutions**:
1. Wait 1 minute and retry
2. Upgrade to paid tier
3. Add retry logic with exponential backoff

### Empty Responses

**Cause**: Safety filters blocking content

**Solution**: Gemini has safety filters for harmful content. Check:
```python
# In src/openai_explain.py, add safety_settings
generation_config={
    "temperature": temperature,
    "max_output_tokens": max_tokens,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
```

---

## API Key Security

### Best Practices

✅ **DO**:
- Store in environment variables
- Use Fly.io secrets for production
- Add to `.gitignore` (already done)
- Rotate keys periodically
- Use separate keys for dev/prod

❌ **DON'T**:
- Commit keys to Git
- Share keys in chat/email
- Hardcode in source files
- Use same key across projects

### If Key is Compromised

1. Go to https://aistudio.google.com/app/apikey
2. Click the 🗑️ icon next to compromised key
3. Create new key
4. Update environment variables immediately

---

## Monitoring Usage

### Check Usage Dashboard

1. Go to https://ai.google.dev/
2. Click "API Keys" in sidebar
3. View usage metrics per key

### Track in Code

```python
# Optional: Add logging
import logging
logging.info(f"Gemini request: {len(input_text)} chars")
```

---

## Migration Guide

### From OpenAI to Gemini

```bash
# Old (OpenAI)
export OPENAI_API_KEY="sk-..."
export FAKESCOPE_LLM_PROVIDER="openai"

# New (Gemini)
export GEMINI_API_KEY="AI..."
export FAKESCOPE_LLM_PROVIDER="gemini"
```

**Code changes**: None! The prompts work identically.

### From Perplexity to Gemini

Same as above—just change the environment variables.

---

## Cost Savings Example

**Scenario**: 1000 fact-check requests/day

| Provider | Daily Cost | Monthly Cost | Yearly Cost |
|----------|-----------|--------------|-------------|
| OpenAI | $10-30 | $300-900 | $3,600-10,800 |
| Perplexity | $10-50 | $300-1,500 | $3,600-18,000 |
| **Gemini (Free)** | **$0** | **$0** | **$0** |
| Gemini (Paid) | $5-10 | $150-300 | $1,800-3,600 |

**Savings**: Up to $10,000/year with Gemini free tier! 💰

---

## FAQs

### Q: Can I use multiple providers?
**A**: Not simultaneously, but you can switch by changing `FAKESCOPE_LLM_PROVIDER`.

### Q: Which is better: Gemini Flash or Pro?
**A**: Flash is better for FakeScope—it's 10x faster and 17x cheaper. Pro is overkill.

### Q: Does Gemini support streaming?
**A**: Yes, but not implemented in current code. Could be added for real-time responses.

### Q: Can I use Gemini with images?
**A**: Yes! Gemini supports image inputs, but FakeScope currently only sends text. Future feature possibility.

### Q: What's the context window size?
**A**: 1M tokens for Flash/Pro (vs. 128K for OpenAI/Perplexity). More than enough.

---

## Next Steps

1. ✅ Get Gemini API key
2. ✅ Set environment variables
3. ✅ Test with sample claim
4. 📊 Monitor usage dashboard
5. 🚀 Deploy to production

**You're all set!** Gemini is now integrated and ready to use.

For questions:
- Gemini Docs: https://ai.google.dev/docs
- FakeScope Issues: https://github.com/enriest/FakeScope/issues
