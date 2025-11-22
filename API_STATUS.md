# FakeScope API Status Summary

## 📊 Current API Status

Generated: November 16, 2025

### ✅ Working APIs
- **Google Gemini**: ✅ API key valid and connected
  - Key: AIzaSyCHVy2sPPiF5-Vc3AqChQpzlgBRm24e0Jw (first 10 chars)
  - Status: Authentication successful
  - Issue: Model name needs verification (tested `gemini-pro` - not found)
  - **Action needed**: Use correct model name like `gemini-1.5-flash` or `gemini-1.5-pro`

### ❌ APIs Needing Attention

#### 1. OpenAI API - INVALID KEY
- Status: ❌ 401 Unauthorized  
- Error: "Incorrect API key provided"
- **Action needed**: Get a new API key from https://platform.openai.com/api-keys
- Current key (expired/invalid): sk-proj-bPkfokm5...c9IA

#### 2. Perplexity API - NOT CONFIGURED
- Status: ❌ No API key set
- **Action needed**: 
  1. Sign up at https://www.perplexity.ai/
  2. Go to https://www.perplexity.ai/settings/api
  3. Create an API key
  4. Add to `.env` file: `PERPLEXITY_API_KEY=pplx-your-key`

---

## 🔧 How to Fix

### Fix #1: Update OpenAI API Key

```bash
# 1. Go to https://platform.openai.com/api-keys
# 2. Click "Create new secret key"
# 3. Copy the key (starts with sk-proj-)
# 4. Update .env file:

export OPENAI_API_KEY="sk-proj-YOUR-NEW-KEY-HERE"
```

### Fix #2: Add Perplexity API Key

```bash
# 1. Sign up at https://www.perplexity.ai/
# 2. Get API key from Settings > API
# 3. Update .env file:

export PERPLEXITY_API_KEY="pplx-YOUR-KEY-HERE"
```

### Fix #3: Verify Gemini Model Name

The Gemini API key works, but we need to use the correct model name. Update `src/openai_explain.py`:

```python
# Change line 11 from:
GEMINI_MODEL = os.getenv("FAKESCOPE_GEMINI_MODEL", "gemini-1.5-flash")

# To one of these verified models:
GEMINI_MODEL = os.getenv("FAKESCOPE_GEMINI_MODEL", "gemini-1.5-flash")  # Fast, recommended
# or
GEMINI_MODEL = os.getenv("FAKESCOPE_GEMINI_MODEL", "gemini-1.5-pro")    # More powerful
```

---

## 🚀 Quick Start After Fixing

Once you have at least ONE working API:

```bash
# 1. Update .env file with new keys
nano .env

# 2. Load environment variables
source .env

# 3. Choose which provider to use
export FAKESCOPE_LLM_PROVIDER=gemini      # Use Gemini (FREE!)
# or
export FAKESCOPE_LLM_PROVIDER=openai      # Use OpenAI (after fixing key)
# or
export FAKESCOPE_LLM_PROVIDER=perplexity  # Use Perplexity (after adding key)

# 4. Test your chosen provider
python test_apis.py

# 5. Run FakeScope
python -m streamlit run src/app.py
```

---

## 💡 Recommended Setup

**For FREE usage (no credit card needed):**
```bash
# Use Gemini - it has a generous free tier
export FAKESCOPE_LLM_PROVIDER=gemini
export GOOGLE_API_KEY="AIzaSyCHVy2sPPiF5-Vc3AqChQpzlgBRm24e0Jw"
export FAKESCOPE_GEMINI_MODEL="gemini-1.5-flash"
```

**For production usage:**
```bash
# Use OpenAI for reliability (requires paid account)
export FAKESCOPE_LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-proj-YOUR-NEW-KEY"
```

**For breaking news/current events:**
```bash
# Use Perplexity for real-time web search
export FAKESCOPE_LLM_PROVIDER=perplexity
export PERPLEXITY_API_KEY="pplx-YOUR-KEY"
```

---

## 📝 Current .env File Contents

Your `.env` file should look like this (with your actual keys):

```bash
# LLM Provider Selection
FAKESCOPE_LLM_PROVIDER=gemini

# OpenAI API (NEEDS NEW KEY)
OPENAI_API_KEY=sk-proj-YOUR-NEW-KEY-HERE
FAKESCOPE_OPENAI_MODEL=gpt-4o-mini

# Google Gemini API (WORKING ✅)
GEMINI_API_KEY=AIzaSyCHVy2sPPiF5-Vc3AqChQpzlgBRm24e0Jw
GOOGLE_API_KEY=AIzaSyCHVy2sPPiF5-Vc3AqChQpzlgBRm24e0Jw
FAKESCOPE_GEMINI_MODEL=gemini-1.5-flash

# Perplexity API (NEEDS KEY)
PERPLEXITY_API_KEY=pplx-YOUR-KEY-HERE
FAKESCOPE_PERPLEXITY_MODEL=llama-3.1-sonar-large-128k-online

# Google Fact Check API (Optional)
GOOGLE_FACTCHECK_API_KEY=your-fact-check-key-here
```

---

## ⚙️ Environment Variable Setup

### Important Note About `GOOGLE_API_KEY` vs `GEMINI_API_KEY`

The Google Generative AI library requires BOTH:
- `GEMINI_API_KEY` - Used by our code (`src/openai_explain.py`)
- `GOOGLE_API_KEY` - Required by google-generativeai library internally

**Both should have the same value!**

This is already configured in your `.env` file.

---

## 🧪 Testing Individual APIs

### Test Gemini Only
```bash
cd "/Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope"
source .env
python -c "
import google.generativeai as genai
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content('Say Gemini works!')
print(response.text)
"
```

### Test OpenAI Only (after fixing key)
```bash
python -c "
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Say OpenAI works!'}],
    max_tokens=10
)
print(response.choices[0].message.content)
"
```

### Test Perplexity Only (after adding key)
```bash
python -c "
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv('PERPLEXITY_API_KEY'), base_url='https://api.perplexity.ai')
response = client.chat.completions.create(
    model='llama-3.1-sonar-large-128k-online',
    messages=[{'role': 'user', 'content': 'Say Perplexity works!'}],
    max_tokens=10
)
print(response.choices[0].message.content)
"
```

---

## 🎯 Next Steps

### Immediate (to get FakeScope working):

1. **Use Gemini for now (it's FREE and working):**
   ```bash
   source .env
   export FAKESCOPE_LLM_PROVIDER=gemini
   python -m streamlit run src/app.py
   ```

2. **Fix OpenAI key (optional, for production):**
   - Visit https://platform.openai.com/api-keys
   - Create new key
   - Update `.env` file

3. **Add Perplexity key (optional, for current events):**
   - Visit https://www.perplexity.ai/settings/api
   - Create new key
   - Update `.env` file

### Testing:
```bash
# Run the comprehensive test
python test_apis.py
```

---

## 📞 Support Resources

- **OpenAI API Keys**: https://platform.openai.com/api-keys
- **Gemini API Keys**: https://ai.google.dev/
- **Perplexity API Keys**: https://www.perplexity.ai/settings/api
- **Gemini Model List**: https://ai.google.dev/models/gemini
- **FakeScope Documentation**: See `GUIDE.md`, `GEMINI_SETUP.md`, `ENHANCED_FEATURES.md`

---

## ✅ Success Criteria

You'll know everything is working when:

1. ✅ `python test_apis.py` shows at least 1/3 providers working
2. ✅ Running `python -m streamlit run src/app.py` starts without errors
3. ✅ You can analyze a claim and see the LLM explanation
4. ✅ No "LLM explanation unavailable" errors in the UI

---

**Bottom Line**: Your Gemini API is already working! You can start using FakeScope right now by running:

```bash
source .env && export FAKESCOPE_LLM_PROVIDER=gemini && python -m streamlit run src/app.py
```

The OpenAI and Perplexity keys just need to be updated/added when you want to use those providers.
