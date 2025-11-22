# FakeScope Hugging Face Spaces - Deployment Complete! 🎉

## Your Space Details

**Space URL**: https://huggingface.co/spaces/enri-est/FakeScope
**Settings URL**: https://huggingface.co/spaces/enri-est/FakeScope/settings
**Build Logs**: https://huggingface.co/spaces/enri-est/FakeScope/logs

## ⚠️ IMPORTANT: Configure Secrets

Your Space needs API keys to function. Please configure them now:

### Step 1: Go to Settings
Visit: https://huggingface.co/spaces/enri-est/FakeScope/settings

### Step 2: Add Repository Secrets

Click on "Repository secrets" and add the following:

#### Required Secrets:

1. **GOOGLE_FACTCHECK_API_KEY**
   - Value: Your Google Fact Check API key
   - Purpose: Fetch external fact-check verification

2. **FAKESCOPE_MODEL_DIR**
   - Value: `enri-est/fakescope-distilbert-2stage`
   - Purpose: Tell FakeScope to load model from your HF Hub

#### LLM Provider (Choose ONE):

**Option A: Gemini (Recommended - Free Tier)**
- **GEMINI_API_KEY**: Your Gemini API key
- **FAKESCOPE_LLM_PROVIDER**: `gemini`

**Option B: OpenAI**
- **OPENAI_API_KEY**: Your OpenAI API key  
- **FAKESCOPE_LLM_PROVIDER**: `openai`

**Option C: Perplexity**
- **PERPLEXITY_API_KEY**: Your Perplexity API key
- **FAKESCOPE_LLM_PROVIDER**: `perplexity`

### Step 3: Save and Restart

After adding secrets:
1. Click "Save"
2. The Space will automatically rebuild (takes ~5-10 minutes)
3. Monitor progress at: https://huggingface.co/spaces/enri-est/FakeScope/logs

## API Keys You Need

### Google Fact Check API
- Get it: https://console.cloud.google.com/
- Enable: "Fact Check Tools API"
- Create: API Key under Credentials

### Gemini API (Recommended)
- Get it: https://ai.google.dev/
- Free tier: 1500 requests/day
- Best value for this project

### OpenAI API (Alternative)
- Get it: https://platform.openai.com/api-keys
- Model: GPT-4o-mini (~$0.01-0.03 per explanation)

### Perplexity API (Alternative)
- Get it: https://www.perplexity.ai/settings/api
- Includes real-time web search

## Testing Your Space

Once secrets are configured and Space is running:

1. **Wait for build** (~5-10 minutes for first deployment)
2. **Visit your Space**: https://huggingface.co/spaces/enri-est/FakeScope
3. **Test with example**:
   - Text: `"Scientists discover breakthrough in renewable energy"`
   - Click "Run Analysis"
   - Verify credibility score, fact checks, and explanation appear

## Troubleshooting

### Space shows "Building..."
- Normal on first deploy
- Check logs: https://huggingface.co/spaces/enri-est/FakeScope/logs
- Takes 5-10 minutes to download dependencies and model

### "Model not found" error
- Check `FAKESCOPE_MODEL_DIR` is set to: `enri-est/fakescope-distilbert-2stage`
- Verify model exists: https://huggingface.co/enri-est/fakescope-distilbert-2stage

### "OpenAI key not configured" warning
- Add appropriate LLM API key (GEMINI_API_KEY, OPENAI_API_KEY, or PERPLEXITY_API_KEY)
- Set `FAKESCOPE_LLM_PROVIDER` to match your provider

### Slow inference
- First load downloads model (~268MB) - takes 1-2 minutes
- Subsequent requests are fast (cached)
- Consider upgrading Space hardware if needed

## Space Features

✅ Automatic HTTPS
✅ Public shareable URL
✅ Built-in Gradio interface
✅ HF Hub model loading (no need to include model in repo)
✅ Environment secrets management
✅ Free hosting for public Spaces

## Next Steps

1. ✅ Configure secrets (see above)
2. ⏳ Wait for build to complete
3. 🧪 Test with sample news articles
4. 🎨 Customize interface in `spaces/app.py` if desired
5. 📢 Share your Space URL!

## Files Deployed

```
enri-est/FakeScope/
├── app.py                 # Gradio interface
├── requirements.txt       # Python dependencies
├── README.md              # Space description
├── .env.example          # Example environment variables
└── src/
    ├── inference.py       # Model loading & prediction
    ├── factcheck.py       # Google Fact Check integration
    ├── openai_explain.py  # LLM explanations
    └── utils.py           # URL extraction utilities
```

## Useful Commands

### Update Space (after making changes)
```bash
cd hf_space_deploy_temp
# Make changes to files
git add .
git commit -m "Update Space"
git push origin main
```

### View Build Logs
```bash
# In browser
https://huggingface.co/spaces/enri-est/FakeScope/logs
```

### Clone Space Locally
```bash
git clone https://huggingface.co/spaces/enri-est/FakeScope
cd FakeScope
# Make changes and push
```

## Support

- **Documentation**: See GUIDE.md in main repository
- **Issues**: https://github.com/enriest/FakeScope/issues
- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces

---

**Deployment Status**: ✅ Complete
**Next Action**: Configure secrets at https://huggingface.co/spaces/enri-est/FakeScope/settings
