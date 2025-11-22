# Deploy FakeScope to Fly.io (Optimized for Large Models)

## Problem
Your Docker image is 8GB+ because it includes the model files. Fly.io has size limits and this makes deployment slow/expensive.

## Solution
Download the model from Hugging Face at runtime instead of copying it into the Docker image.

## Step 1: Upload Model to Hugging Face (If not already done)

```bash
# Login to Hugging Face
huggingface-cli login

# Upload your model (replace YOUR_USERNAME with your HF username)
python src/upload_model_hg.py \
  --repo-id YOUR_USERNAME/distilbert-fakenews-2stage \
  --model-dir ./distilbert_fakenews_2stage

# For private repos, add --private flag
```

## Step 2: Set Environment Variables in Fly.io

```bash
# Set the model to download from HF (replace with your actual repo)
flyctl secrets set FAKESCOPE_MODEL_DIR="YOUR_USERNAME/distilbert-fakenews-2stage" --app fakescope

# If your HF repo is private, add your token
flyctl secrets set HF_TOKEN="your_huggingface_token_here" --app fakescope

# Keep existing API keys
# (they should already be set: OPENAI_API_KEY, GOOGLE_FACTCHECK_API_KEY, GEMINI_API_KEY)
```

## Step 3: Deploy

```bash
# Deploy with increased timeout for initial model download
flyctl deploy --app fakescope --wait-timeout 600

# The first deployment will take ~5-10 minutes to download the model
# Subsequent deploys will be faster due to volume caching
```

## Step 4: Monitor Deployment

```bash
# Watch logs during deployment
flyctl logs --app fakescope

# Check machine status
flyctl status --app fakescope

# SSH into machine if needed
flyctl ssh console --app fakescope
```

## Benefits of This Approach

1. **Small Docker Image**: ~500MB instead of 8GB+
2. **Faster Builds**: No need to upload 8GB on every deploy
3. **Version Control**: Change models by updating env var, no rebuild needed
4. **Cost Efficient**: Less bandwidth and storage usage

## Troubleshooting

### Model Download Fails
- Check `HF_TOKEN` is set if repo is private
- Verify `FAKESCOPE_MODEL_DIR` format: `username/repo-name`
- Check logs: `flyctl logs --app fakescope`

### Machine Timeout on First Start
- Model download can take 5-10 minutes first time
- Use `--wait-timeout 600` flag
- Check machine has enough memory (2GB+ recommended)

### Persistent Storage
The model will be cached on the `/data` volume mount after first download, making subsequent restarts faster.

## Machine Configuration

Current setup in `fly.toml`:
- Memory: Check with `flyctl scale show`
- Recommended: 2GB RAM minimum for model inference
- Increase if needed: `flyctl scale memory 2048`
