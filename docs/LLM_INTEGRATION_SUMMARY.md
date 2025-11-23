# LLM Integration Summary - Perplexity & OpenAI

## ✅ What Was Added

Your FakeScope project now supports **both OpenAI and Perplexity APIs** for LLM explanations!

### 1. **Updated Files**

#### `src/openai_explain.py` (Main Changes)
- Added support for Perplexity API alongside OpenAI
- New environment variables:
  - `FAKESCOPE_LLM_PROVIDER` - Choose "openai" or "perplexity" (default: "openai")
  - `PERPLEXITY_API_KEY` - Your Perplexity API key
  - `FAKESCOPE_PERPLEXITY_MODEL` - Model selection (default: "llama-3.1-sonar-large-128k-online")
- Both providers use the same prompts (OpenAI-compatible API)
- Clear comments showing where to customize prompts

#### `src/app.py`
- Updated error message to mention both API options

#### `GUIDE.md`
- Added Perplexity API key setup instructions
- Updated Docker run commands with new env vars
- Updated Fly.io secrets configuration
- Updated Hugging Face Spaces configuration
- Added comprehensive **"Customizing LLM Prompts"** section
- Updated cost estimates to include Perplexity

#### `README.md`
- Added Perplexity API key setup
- Added FAQ about choosing between OpenAI and Perplexity
- Added FAQ about where to find prompts
- Updated cost estimates

#### **New File**: `PROMPT_CUSTOMIZATION.md`
- Complete guide to customizing prompts
- Examples of different prompt patterns
- Testing procedures
- Provider-specific tips
- Troubleshooting guide

---

## 🔧 How to Use It

### Option 1: Use OpenAI (Default)
```bash
export OPENAI_API_KEY="sk-your-key-here"
export FAKESCOPE_LLM_PROVIDER="openai"  # Optional, this is default
```

### Option 2: Use Perplexity
```bash
export PERPLEXITY_API_KEY="pplx-your-key-here"
export FAKESCOPE_LLM_PROVIDER="perplexity"
```

### Run the App
```bash
source .venv/bin/activate
python -m streamlit run src/app.py
```

---

## 📍 Where to Find & Modify Prompts

**File**: `src/openai_explain.py`  
**Function**: `generate_explanation()`  
**Lines**: ~75-95

### Two Prompts to Customize:

#### 1. System Prompt (Line ~83)
```python
system_prompt = "You are an expert, neutral fact-checking assistant. Be precise and cite sources."
```
**What it does**: Defines the AI's personality and role

#### 2. User Prompt (Lines ~85-92)
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

### Example: Make Output More Concise
Change user_prompt to:
```python
user_prompt = (
    "In 2-3 sentences, explain if this claim is true or false:\n\n"
    f"{_truncate(input_text, 500)}\n\n"
    f"Model prediction: {'FAKE' if model_scores.get('fake') > 0.5 else 'TRUE'}\n"
    f"External evidence: {evidence_str}"
)
```

### Example: Add Confidence Ratings
```python
user_prompt = (
    "Rate this claim's credibility 0-100 and explain:\n\n"
    f"Claim: {_truncate(input_text)}\n"
    f"Model: {model_scores.get('true'):.0%} likely true\n"
    f"Evidence: {evidence_str}\n\n"
    "Format:\n"
    "Score: [0-100]\n"
    "Reasoning: [brief explanation]"
)
```

For **many more examples**, see `PROMPT_CUSTOMIZATION.md`!

---

## 🚀 Quick Test

Test your Perplexity integration:

```bash
# Set your API key
export PERPLEXITY_API_KEY="pplx-your-key-here"
export FAKESCOPE_LLM_PROVIDER="perplexity"

# Create test script
cat > test_perplexity.py << 'EOF'
from src.openai_explain import generate_explanation

result = generate_explanation(
    input_text="Scientists discover new renewable energy breakthrough",
    model_scores={"fake": 0.3, "true": 0.7},
    google_items=[],
    google_score=None
)

print(result)
EOF

# Run test
python test_perplexity.py
```

Expected output: A fact-checking explanation from Perplexity.

---

## 📊 Choosing Between OpenAI and Perplexity

| Feature | OpenAI (GPT-4o-mini) | Perplexity (Sonar) |
|---------|---------------------|-------------------|
| **Speed** | ⚡ Very fast (1-2s) | 🟢 Fast (2-4s) |
| **Cost** | 💰 ~$0.01-0.03/explanation | 💰 ~$0.01-0.05/explanation |
| **Real-time search** | ❌ No | ✅ Yes (with "online" models) |
| **Best for** | Concise, structured output | Current events, finding sources |
| **Consistency** | ⭐ Very high | 🟢 Good |
| **Setup** | Easy (well-documented) | Easy (OpenAI-compatible) |

**Recommendation**: 
- Use **OpenAI** for general fact-checking (faster, cheaper, more consistent)
- Use **Perplexity** when checking recent news or need real-time web sources

---

## 🔐 Environment Variables Reference

```bash
# Required: Choose ONE LLM provider
export OPENAI_API_KEY="sk-..."              # For OpenAI
# OR
export PERPLEXITY_API_KEY="pplx-..."        # For Perplexity

# Optional: Select provider (default: openai)
export FAKESCOPE_LLM_PROVIDER="openai"      # or "perplexity"

# Optional: Customize models
export FAKESCOPE_OPENAI_MODEL="gpt-4o-mini"        # Default OpenAI model
export FAKESCOPE_PERPLEXITY_MODEL="llama-3.1-sonar-large-128k-online"  # Default Perplexity

# Other APIs
export GOOGLE_FACTCHECK_API_KEY="your-key"  # For external fact-checking
```

---

## 📦 Deployment Checklist

### Local Testing
- [x] Set `PERPLEXITY_API_KEY` environment variable
- [x] Set `FAKESCOPE_LLM_PROVIDER="perplexity"`
- [x] Run `streamlit run src/app.py`
- [x] Test with a claim, verify explanation appears

### Docker
```bash
docker build -t fakescope:latest .

docker run --rm -p 8080:8080 \
  -e PERPLEXITY_API_KEY="$PERPLEXITY_API_KEY" \
  -e FAKESCOPE_LLM_PROVIDER="perplexity" \
  -e GOOGLE_FACTCHECK_API_KEY="$GOOGLE_FACTCHECK_API_KEY" \
  fakescope:latest
```

### Fly.io
```bash
# Set secrets
flyctl secrets set PERPLEXITY_API_KEY="pplx-your-key"
flyctl secrets set FAKESCOPE_LLM_PROVIDER="perplexity"

# Deploy
flyctl deploy
```

### Hugging Face Spaces
1. Go to Space → Settings → Repository secrets
2. Add:
   - `PERPLEXITY_API_KEY`: `pplx-your-key`
   - `FAKESCOPE_LLM_PROVIDER`: `perplexity`
3. Push code changes

---

## 📚 Documentation Files

1. **`GUIDE.md`** - Full deployment guide (updated with Perplexity)
2. **`PROMPT_CUSTOMIZATION.md`** - NEW! Complete prompt customization guide
3. **`LLM_INTEGRATION_SUMMARY.md`** - This file (quick reference)
4. **`README.md`** - Updated with Perplexity info

---

## 🐛 Troubleshooting

### "LLM explanation unavailable"
**Cause**: API key not set or provider misconfigured

**Solution**:
```bash
# Check if key is set
echo $PERPLEXITY_API_KEY

# Check provider
echo $FAKESCOPE_LLM_PROVIDER

# Re-export if needed
export PERPLEXITY_API_KEY="pplx-..."
export FAKESCOPE_LLM_PROVIDER="perplexity"
```

### "API error: Invalid API key"
**Cause**: Wrong API key format or expired key

**Solution**: 
- Verify key starts with `pplx-` for Perplexity
- Check key at https://www.perplexity.ai/settings/api
- Generate new key if expired

### Slow responses
**Cause**: Perplexity "online" models do web search (takes longer)

**Solution**: Use non-online model for speed
```bash
export FAKESCOPE_PERPLEXITY_MODEL="llama-3.1-70b-instruct"
```

---

## 🎯 Next Steps

1. **Test both providers**: Try OpenAI and Perplexity, compare results
2. **Customize prompts**: Edit `src/openai_explain.py` to match your needs
3. **Monitor costs**: Track API usage in dashboards
4. **Deploy**: Push to Fly.io or Hugging Face Spaces

---

## 💡 Examples of What You Can Do

### Switch providers on the fly
```bash
# Morning: Use OpenAI for quick checks
export FAKESCOPE_LLM_PROVIDER="openai"
streamlit run src/app.py

# Afternoon: Use Perplexity for recent news
export FAKESCOPE_LLM_PROVIDER="perplexity"
streamlit run src/app.py
```

### A/B test prompt changes
```python
# In src/openai_explain.py, try two versions:

# Version A: Concise
user_prompt = f"Is this fake? {_truncate(input_text, 200)}"

# Version B: Detailed
user_prompt = f"Analyze in detail:\n{_truncate(input_text, 2000)}\n\nEvidence:\n{evidence_str}"
```

### Custom model selection
```bash
# Try GPT-4 for better quality (more expensive)
export FAKESCOPE_OPENAI_MODEL="gpt-4o"

# Try smaller Perplexity model (faster, cheaper)
export FAKESCOPE_PERPLEXITY_MODEL="llama-3.1-sonar-small-128k-online"
```

---

**🎉 You're all set!** Your FakeScope now supports both OpenAI and Perplexity with full prompt customization.

For questions or issues, check:
- `PROMPT_CUSTOMIZATION.md` - Detailed prompt guide
- `GUIDE.md` - Full deployment guide
- GitHub Issues: https://github.com/enriest/FakeScope/issues
