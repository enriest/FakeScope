# Gemini API Debugging Guide

## ✅ Changes Made

### 1. **Updated Model Names (2025 Current)**
All Gemini model references updated from deprecated 1.5 versions to current 2025 models:

**Files Updated:**
- `hf_streamlit_deploy/src/app.py`
- `hf_streamlit_deploy/src/openai_explain.py`
- `src/app.py`
- `src/app_enhanced.py`
- `src/openai_explain.py`

**Model Migration:**
```python
# OLD (Deprecated)
["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-1.5-flash-8b"]

# NEW (Current 2025)
["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]
```

### 2. **Added Comprehensive Logging**
Enhanced `hf_streamlit_deploy/src/app.py` with detailed debug logging:

**Startup Logging:**
```python
# Logs at app startup
logger.info("FakeScope API Configuration Check")
logger.info(f"OPENAI_API_KEY present: {bool(...)}")
logger.info(f"GEMINI_API_KEY present: {bool(...)}")
logger.info(f"GEMINI_API_KEY length: {len(key)} chars, prefix: {key[:10]}...")
```

**Runtime Logging (Per Request):**
```python
logger.info(f"[GEMINI] Attempting Gemini API call")
logger.info(f"[GEMINI] API key present: {bool(gemini_key)}")
logger.info(f"[GEMINI] Will try models in order: {models_to_try}")
logger.info(f"[GEMINI] Trying model: {model_name}")
logger.info(f"[GEMINI] Response status: {resp.status_code}")
logger.info(f"[GEMINI] ✅ Success with {model_name}")
logger.error(f"[GEMINI] ❌ All models failed")
```

### 3. **Dependencies Verified**
**`hf_streamlit_deploy/requirements.txt`:**
```txt
google-generativeai>=0.8.0,<1.0.0  ✅
requests>=2.32.3                    ✅
```

**`requirements.txt` (main):**
```txt
google-generativeai>=0.8.0,<1.0.0  ✅
```

### 4. **Rate Limit Handling**
Added exception handling for both OpenAI and Gemini:
```python
except openai.RateLimitError as e:
    return f"⏳ {provider.title()} API rate limit reached..."
except openai.APIError as e:
    return f"❌ {provider.title()} API error: {str(e)}"
```

## 🧪 Local Testing

### Run the Test Script
```bash
# Activate your environment
source .venv/bin/activate

# Run the test
python test_gemini_local.py
```

The script tests:
1. **REST API** (same as production HF deployment)
2. **Official SDK** (for comparison)
3. **Model availability** (lists all accessible models)

### What the Test Checks
- ✅ GEMINI_API_KEY environment variable presence
- ✅ API key format and length
- ✅ All three fallback models (gemini-2.5-flash, 2.0-flash, 2.5-flash-lite)
- ✅ Safety settings configuration
- ✅ Response structure and content
- ✅ Error messages and HTTP codes

### Expected Output
```
============================================================
FakeScope Gemini API Local Test
============================================================
✅ GEMINI_API_KEY found (39 chars)
   Prefix: AIzaSyAbc1...

Testing model: gemini-2.5-flash
------------------------------------------------------------
   Making request to: https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent
   Status code: 200
   Response keys: ['candidates', 'usageMetadata', 'modelVersion']
   Candidates: 1
   ✅ SUCCESS! Response length: 45 chars
   Response preview: API connection successful...
```

## 🔍 HuggingFace Deployment Debugging

### Check Deployment Logs
After deploying with the new logging, check HF Spaces logs for:

```
[GEMINI] Attempting Gemini API call
[GEMINI] API key present: True
[GEMINI] API key length: 39 chars
[GEMINI] Will try models in order: ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.5-flash-lite']
[GEMINI] Trying model: gemini-2.5-flash
[GEMINI] Request URL: https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent
[GEMINI] Response status: 200
[GEMINI] ✅ Success with gemini-2.5-flash, response length: 245 chars
```

### Common Issues & Solutions

#### Issue 1: API Key Not Found
**Log Pattern:**
```
[GEMINI] GEMINI_API_KEY not found in environment
```

**Solution:**
1. Go to HF Space Settings → Variables
2. Add `GEMINI_API_KEY` with your key from https://aistudio.google.com/apikey
3. Restart the Space

#### Issue 2: Model Not Found (404)
**Log Pattern:**
```
[GEMINI] HTTP 404 error with gemini-2.5-flash: models/gemini-2.5-flash is not found
```

**Solution:**
- The model may not be available in your region
- Check available models at: https://ai.google.dev/gemini-api/docs/models/gemini
- The fallback mechanism will try alternative models

#### Issue 3: Safety Filter Blocking
**Log Pattern:**
```
[GEMINI] No candidates from gemini-2.5-flash. Full response: {'promptFeedback': {'blockReason': 'SAFETY'}}
```

**Solution:**
- Already configured with `BLOCK_NONE` for all safety categories
- If still blocked, the content may violate Google's terms
- Try rephrasing the input or use OpenAI/Perplexity instead

#### Issue 4: Rate Limiting
**Log Pattern:**
```
[GEMINI] HTTP 429 - Resource has been exhausted
```

**Solution:**
- Free tier has limits: 15 RPM, 1 million TPM, 1500 RPD
- Upgrade to paid tier or implement request queuing
- Consider using OpenAI as primary with Gemini as fallback

## 📊 Verification Checklist

### Local Environment
- [ ] Run `test_gemini_local.py` successfully
- [ ] Verify all 3 models are accessible
- [ ] Test with actual fake news content

### HuggingFace Space
- [ ] Environment variable `GEMINI_API_KEY` is set
- [ ] Deployment logs show `[GEMINI]` startup messages
- [ ] API key presence logged as `True`
- [ ] Chat requests show model attempts in logs

### Production Testing
1. Deploy to HF Spaces
2. Check logs for startup configuration
3. Try Chat & Debate tab with Gemini selected
4. Verify logs show successful API calls
5. Test fallback behavior by using invalid key temporarily

## 🔧 Alternative Solutions

### If Gemini Still Fails in Production

**Option 1: Use OpenAI as Primary**
- OpenAI is more reliable in production
- Better rate limits on free tier
- More consistent responses

**Option 2: Use Perplexity**
- Good alternative to Gemini
- Better for fact-checking tasks
- Built-in web search capability

**Option 3: SDK vs REST API**
If REST API fails, try switching to official SDK:

```python
# In openai_explain.py
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')
response = model.generate_content(prompt)
```

## 📝 Next Steps

1. **Test Locally:**
   ```bash
   python test_gemini_local.py
   ```

2. **Deploy Changes:**
   ```bash
   git add .
   git commit -m "Add Gemini debugging and update models to 2025 versions"
   git push
   ```

3. **Monitor HF Logs:**
   - Check for `[GEMINI]` log entries
   - Verify API key is present
   - Confirm model attempts

4. **Test in Production:**
   - Go to Chat & Debate tab
   - Select Gemini as provider
   - Try a sample conversation
   - Check logs for success/failure

## 📚 References

- **Gemini Models (2025):** https://ai.google.dev/gemini-api/docs/models/gemini
- **Gemini API Docs:** https://ai.google.dev/gemini-api/docs
- **API Keys:** https://aistudio.google.com/apikey
- **Rate Limits:** https://ai.google.dev/gemini-api/docs/rate-limits
- **Troubleshooting:** https://ai.google.dev/gemini-api/docs/troubleshooting

---

**Created:** November 22, 2025  
**Last Updated:** November 22, 2025
