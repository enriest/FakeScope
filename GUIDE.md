# FakeScope Deployment Guide – Step by Step

This guide walks you through deploying FakeScope from scratch to a live web application. We'll cover local testing, Docker containerization, and production deployment to Fly.io or Hugging Face Spaces.

---

## Table of Contents

1. [Prerequisites & Setup](#1-prerequisites--setup)
2. [Local Development & Testing](#2-local-development--testing)
3. [Docker Build & Local Testing](#3-docker-build--local-testing)
4. [Fly.io Deployment (Recommended)](#4-flyio-deployment-recommended)
5. [Hugging Face Spaces Alternative](#5-hugging-face-spaces-alternative)
6. [Post-Deployment Testing](#6-post-deployment-testing)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites & Setup

### 1.1 System Requirements

**Purpose**: Ensure your development environment has all necessary tools.

- **Operating System**: macOS, Linux, or Windows with WSL2
- **Python**: 3.9+ (3.11 recommended)
- **Docker**: Latest version from [docker.com](https://www.docker.com/get-started)
- **Git**: For version control

**Verify installations**:
```bash
python3 --version  # Should show 3.9 or higher
docker --version   # Should show Docker version
git --version      # Should show Git version
```

### 1.2 Account Setup

**Purpose**: Create accounts on deployment platforms.

1. **Fly.io Account** (Primary deployment):
   - Visit https://fly.io/app/sign-up
   - Sign up with GitHub or email
   - Install `flyctl` CLI:
     ```bash
     # macOS/Linux
     curl -L https://fly.io/install.sh | sh
     
     # Add to PATH (add to ~/.zshrc or ~/.bashrc)
     export PATH="$HOME/.fly/bin:$PATH"
     ```
   - Verify installation:
     ```bash
     flyctl version
     ```
   - Login:
     ```bash
     flyctl auth login
     ```

2. **Hugging Face Account** (Alternative deployment):
   - Visit https://huggingface.co/join
   - Create account
   - Generate access token: Settings → Access Tokens → New token

### 1.3 API Keys Setup

**Purpose**: Obtain API keys for external services (OpenAI and Google Fact Check).

#### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign in or create account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. **Important**: Store securely - you won't see it again

#### Google Fact Check API Key
1. Go to https://console.cloud.google.com/
2. Create a new project or select existing
3. Enable "Fact Check Tools API":
   - Navigate to "APIs & Services" → "Library"
   - Search for "Fact Check Tools API"
   - Click "Enable"
4. Create credentials:
   - "APIs & Services" → "Credentials"
   - "Create Credentials" → "API Key"
   - Copy the API key

**Store keys temporarily**:
```bash
# Add to your ~/.zshrc or ~/.bashrc for easy access
export OPENAI_API_KEY="sk-your-openai-key-here"
export GOOGLE_FACTCHECK_API_KEY="your-google-key-here"

# Reload shell
source ~/.zshrc
```

---

## 2. Local Development & Testing

**Purpose**: Verify the application works on your machine before containerizing.

### 2.1 Clone Repository

```bash
cd ~/Projects  # or your preferred directory
git clone https://github.com/enriest/FakeScope.git
cd FakeScope
```

### 2.2 Create Virtual Environment

**Purpose**: Isolate project dependencies from system Python.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2.3 Install Dependencies

**Purpose**: Install all required Python packages.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected time**: 5-10 minutes (downloads ~2GB of packages including PyTorch)

### 2.4 Verify Model Exists

**Purpose**: Ensure the trained DistilBERT model is present.

```bash
ls -lh distilbert_fakenews_2stage/
```

**Expected output**:
```
config.json
model.safetensors  (~268MB)
tokenizer.json
vocab.txt
...
```

**If model is missing**: You need to train it first using `Project.ipynb` or download from your storage.

### 2.5 Test Inference Module

**Purpose**: Verify model loading and prediction work.

```bash
python3 -c "
from src.inference import credibility_score
score = credibility_score('Scientists discover new renewable energy breakthrough')
print(f'Credibility: {score:.1f}/100')
"
```

**Expected**: Should print a credibility score without errors.

### 2.6 Test Streamlit UI Locally

**Purpose**: Run the web interface on your machine.

```bash
streamlit run src/app.py
```

**What happens**:
- Streamlit starts on http://localhost:8501 (default)
- Opens automatically in your browser
- You should see the FakeScope interface with two tabs

**Test the app**:
1. Go to "Predict" tab
2. Enter text: `"Breaking news: Scientists cure cancer with revolutionary treatment"`
3. Click "Run Analysis"
4. Should see model score, Google fact checks (if API key set), and explanation

Press `Ctrl+C` in terminal to stop.

### 2.7 Test FastAPI Backend

**Purpose**: Verify the API endpoint works.

**Terminal 1** - Start API:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

**Terminal 2** - Test healthcheck:
```bash
curl http://localhost:8001/healthz
```
**Expected**: `{"status":"ok"}`

**Test prediction**:
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Scientists discover breakthrough in renewable energy","include_factcheck":false}'
```

**Expected**: JSON with credibility, probs, google_score

Press `Ctrl+C` in Terminal 1 to stop.

---

## 3. Docker Build & Local Testing

**Purpose**: Package the application into a container that can run anywhere.

### 3.1 Understand Docker Context

**What gets included**:
- `src/` - Application code
- `distilbert_fakenews_2stage/` - Trained model (~268MB)
- `requirements.txt` - Dependencies

**What gets excluded** (via `.dockerignore`):
- `datasets/` - Raw training data (not needed for inference)
- `.git/` - Git history
- `*.ipynb` - Jupyter notebooks
- `mlruns/`, `results*/` - Training artifacts

### 3.2 Build Docker Image

**Purpose**: Create a runnable container image.

```bash
docker build -t fakescope:latest .
```

**What happens**:
1. Pulls Python 3.11 base image
2. Installs system dependencies (build tools, XML parsers)
3. Installs Python packages from `requirements.txt`
4. Copies application code and model
5. Sets up entry point to run both Streamlit and FastAPI

**Expected time**: 10-15 minutes on first build

**Expected output**: Should end with `Successfully tagged fakescope:latest`

**Check image size**:
```bash
docker images fakescope:latest
```
**Expected**: ~4-5 GB (PyTorch + transformers are large)

### 3.3 Run Container Locally

**Purpose**: Test the containerized application before deploying.

```bash
docker run --rm -p 8080:8080 -p 8001:8001 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e GOOGLE_FACTCHECK_API_KEY="$GOOGLE_FACTCHECK_API_KEY" \
  fakescope:latest
```

**What this does**:
- `--rm` - Remove container when stopped
- `-p 8080:8080` - Map Streamlit port (UI)
- `-p 8001:8001` - Map FastAPI port (API)
- `-e` - Pass environment variables (API keys)

**Access the app**:
- **Streamlit UI**: http://localhost:8080
- **API healthcheck**: http://localhost:8001/healthz
- **API docs**: http://localhost:8001/docs (Swagger UI)

**Test thoroughly**:
1. Submit an article URL
2. Check model prediction appears
3. Verify Google fact check results show (if key set)
4. Confirm explanation is generated (if OpenAI key set)
5. Go to "Dashboard" tab - should see your submission

Press `Ctrl+C` to stop container.

---

## 4. Fly.io Deployment (Recommended)

**Purpose**: Deploy to production with persistent storage and automatic HTTPS.

### 4.1 Initialize Fly.io App

**Purpose**: Create app configuration without deploying yet.

```bash
flyctl launch --no-deploy
```

**Prompts**:
- **App name**: Choose unique name (e.g., `fakescope-yourname`) or let Fly generate
- **Region**: Choose closest to your users (e.g., `iad` for US East)
- **PostgreSQL**: No (we use SQLite)
- **Redis**: No

**What this creates**:
- Updates `fly.toml` with your app name
- Registers app on Fly.io

### 4.2 Create Persistent Volume

**Purpose**: Store SQLite database that persists across deployments.

```bash
flyctl volumes create fakescope_data --size 1 --region iad
```

**Parameters**:
- `fakescope_data` - Volume name (must match `fly.toml` mount source)
- `--size 1` - 1 GB (sufficient for thousands of predictions)
- `--region iad` - Must match app region

**Why this matters**: Without a volume, database resets on each deploy. The volume ensures your dashboard history persists.

### 4.3 Set Secrets

**Purpose**: Securely store API keys (not in code or logs).

```bash
flyctl secrets set OPENAI_API_KEY="$OPENAI_API_KEY"
flyctl secrets set GOOGLE_FACTCHECK_API_KEY="$GOOGLE_FACTCHECK_API_KEY"
```

**Verify secrets are set**:
```bash
flyctl secrets list
```

**Expected output**:
```
NAME                         DIGEST                  CREATED AT
OPENAI_API_KEY               abc123...               2024-01-15
GOOGLE_FACTCHECK_API_KEY     def456...               2024-01-15
```

### 4.4 Deploy to Fly.io

**Purpose**: Build and deploy your app to production.

```bash
flyctl deploy
```

**What happens**:
1. Uploads Docker context to Fly.io
2. Builds Docker image remotely
3. Pushes to Fly.io registry
4. Creates VM and starts container
5. Mounts volume at `/data`
6. Injects secrets as environment variables
7. Starts Streamlit (port 8080) and FastAPI (port 8001)
8. Configures HTTPS certificate automatically

**Expected time**: 15-20 minutes on first deploy

**Expected output**:
```
...
==> Monitoring deployment

✓ [app] completed
--> v0 deployed successfully
```

### 4.5 Get Your App URL

```bash
flyctl info
```

**Look for**:
```
Hostname: fakescope-yourname.fly.dev
```

**Access your live app**: https://fakescope-yourname.fly.dev

### 4.6 View Logs (Optional)

**Purpose**: Monitor application health and debug issues.

```bash
# Stream live logs
flyctl logs

# Show recent logs
flyctl logs --limit 100
```

---

## 5. Hugging Face Spaces Alternative

**Purpose**: Deploy using Hugging Face's Gradio-based platform (simpler but less control).

### 5.1 Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. **Settings**:
   - **Name**: `fakescope` (or your choice)
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free) or upgrade for faster inference

### 5.2 Prepare Files

**Purpose**: Gather files needed for Spaces.

```bash
# In your FakeScope directory
mkdir -p hf_space_deploy
cd hf_space_deploy

# Copy Gradio app
cp ../spaces/app.py .
cp ../spaces/requirements.txt .
cp ../spaces/README.md .

# Copy source modules
mkdir src
cp ../src/inference.py src/
cp ../src/factcheck.py src/
cp ../src/openai_explain.py src/
cp ../src/utils.py src/
touch src/__init__.py

# Copy model (large - may take a minute)
cp -r ../distilbert_fakenews_2stage .
```

### 5.3 Push to Space

**Option A: Git CLI**
```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/fakescope
cd fakescope

# Copy files
cp -r ../hf_space_deploy/* .

# Commit and push
git add .
git commit -m "Initial FakeScope deployment"
git push
```

**Option B: Web UI**
1. In Space settings, use "Files and versions" tab
2. Click "Add file" → "Upload files"
3. Upload all files from `hf_space_deploy/`
4. **Note**: For large models, you may need Git LFS:
   ```bash
   git lfs install
   git lfs track "*.safetensors"
   git add .gitattributes
   ```

### 5.4 Configure Secrets

**Purpose**: Add API keys to Space environment.

1. In your Space, click "Settings"
2. Scroll to "Repository secrets"
3. Add two secrets:
   - **Name**: `OPENAI_API_KEY`, **Value**: your OpenAI key
   - **Name**: `GOOGLE_FACTCHECK_API_KEY`, **Value**: your Google key
4. Click "Save"

### 5.5 Wait for Build

**What happens**:
- Space automatically builds on push
- Installs dependencies from `requirements.txt`
- Loads model
- Starts Gradio interface

**Check build status**: Look for green checkmark in Space UI

**Access your Space**: https://huggingface.co/spaces/YOUR_USERNAME/fakescope

---

## 6. Post-Deployment Testing

**Purpose**: Verify production deployment works correctly.

### 6.1 Test Streamlit UI (Fly.io)

1. Open `https://your-app.fly.dev`
2. **Test text input**:
   - Enter: `"Local officials announce new infrastructure project"`
   - Click "Run Analysis"
   - Verify credibility score appears
3. **Test URL input**:
   - Enter: `https://www.bbc.com/news` (or any news article)
   - Click "Fetch text from URL"
   - Should extract article text
   - Run analysis
4. **Check dashboard**:
   - Go to "Dashboard" tab
   - Should see your recent analyses
   - Chart should show score trend

### 6.2 Test API Endpoint (Fly.io)

**Note**: By default, only port 8080 (Streamlit) is exposed. To test API, either:
- SSH into Fly VM: `flyctl ssh console` then `curl localhost:8001/healthz`
- Or expose port 8001 publicly (see Troubleshooting section)

### 6.3 Test Gradio Interface (HF Spaces)

1. Open your Space URL
2. Fill in form:
   - **Article URL**: (optional) `https://www.reuters.com/world/`
   - **Title**: (optional) `"New climate agreement reached"`
   - **Text**: `"World leaders signed a historic climate agreement today"`
3. Click "Submit"
4. Verify results table shows:
   - Credibility score
   - Probabilities
   - Google score
   - Fact check sources
   - Explanation text

### 6.4 Verify Persistence (Fly.io)

**Purpose**: Ensure dashboard history survives redeployments.

1. Make several predictions via UI
2. Check Dashboard tab - note prediction count
3. Redeploy:
   ```bash
   flyctl deploy
   ```
4. After deploy completes, refresh UI
5. Dashboard should still show previous predictions

**Why this works**: Volume at `/data` persists across deployments.

---

## 7. Troubleshooting

### Common Issues

#### Issue: Model not found error
```
FileNotFoundError: Model not found at './distilbert_fakenews_2stage'
```

**Cause**: Model directory missing from Docker context

**Solution**:
```bash
# Verify model exists locally
ls -lh distilbert_fakenews_2stage/

# Check .dockerignore doesn't exclude model
grep distilbert .dockerignore  # Should NOT see distilbert_fakenews_2stage

# Rebuild with verbose output
docker build -t fakescope:latest . --progress=plain
```

#### Issue: Out of memory on Fly.io
```
Error: killed (Out of memory)
```

**Cause**: Default VM size (256MB) too small for PyTorch

**Solution**: Scale to larger VM
```bash
flyctl scale memory 2048  # 2GB
flyctl scale vm shared-cpu-2x  # 2x CPU
```

#### Issue: API keys not working
```
Warning: OpenAI key not configured
```

**Cause**: Secrets not loaded properly

**Solution**:
```bash
# Verify secrets exist
flyctl secrets list

# Re-set if missing
flyctl secrets set OPENAI_API_KEY="$OPENAI_API_KEY"

# Check logs for errors
flyctl logs
```

#### Issue: Volume mount failed
```
Error: no volumes available
```

**Cause**: Volume not created or wrong region

**Solution**:
```bash
# List volumes
flyctl volumes list

# Create in correct region
flyctl volumes create fakescope_data --size 1 --region iad

# Update fly.toml region to match
flyctl regions set iad
```

#### Issue: Database locked
```
sqlite3.OperationalError: database is locked
```

**Cause**: Multiple processes writing simultaneously (rare)

**Solution**: Streamlit and FastAPI use separate connections. If persistent:
```bash
# SSH into VM
flyctl ssh console

# Check DB permissions
ls -lh /data/predictions.db

# Remove and let app recreate
rm /data/predictions.db
exit

# Restart app
flyctl apps restart fakescope-yourname
```

### Expose FastAPI Publicly (Optional)

**Purpose**: Make API accessible at `https://your-app.fly.dev:8001/predict`

Edit `fly.toml`:
```toml
# Add after existing [[services]] block
[[services]]
  internal_port = 8001
  processes = ["app"]
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 8001

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 8001
```

Redeploy:
```bash
flyctl deploy
```

Test:
```bash
curl https://your-app.fly.dev:8001/healthz
```

### Enable Debug Logging

**Fly.io**:
```bash
# Set debug env var
flyctl secrets set DEBUG=1

# View detailed logs
flyctl logs
```

**Local Docker**:
```bash
docker run --rm -p 8080:8080 \
  -e DEBUG=1 \
  fakescope:latest
```

### Health Check Adjustments

If health checks fail but app works, adjust in `fly.toml`:
```toml
[[services.checks]]
  name = "web"
  type = "http"
  interval = "30s"      # Increase from 15s
  timeout = "5s"        # Increase from 2s
  path = "/"
  grace_period = "60s"  # Add grace period for model loading
```

---

## Quick Reference

### Fly.io Commands Cheat Sheet
```bash
# Deploy
flyctl deploy

# View logs
flyctl logs

# SSH into VM
flyctl ssh console

# Scale resources
flyctl scale memory 2048
flyctl scale vm shared-cpu-2x

# Restart app
flyctl apps restart

# Open in browser
flyctl open

# View secrets
flyctl secrets list

# View volumes
flyctl volumes list

# Delete app (careful!)
flyctl apps destroy fakescope-yourname
```

### Local Testing Commands
```bash
# Run Streamlit
streamlit run src/app.py

# Run FastAPI
uvicorn src.api:app --reload --port 8001

# Run both with Docker
docker run --rm -p 8080:8080 -p 8001:8001 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e GOOGLE_FACTCHECK_API_KEY="$GOOGLE_FACTCHECK_API_KEY" \
  fakescope:latest

# Test API
curl http://localhost:8001/healthz
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Test article","include_factcheck":false}'
```

---

## Cost Estimates

### Fly.io (Pay as you go)
- **Free tier**: Includes 3 shared-cpu-1x VMs (256MB) - **likely insufficient**
- **Recommended**: shared-cpu-2x (2GB) - ~$10-15/month
- **Volume**: 1GB - $0.15/month
- **Bandwidth**: 100GB free, then $0.02/GB

**Estimated monthly cost**: $10-20 for low-medium traffic

### Hugging Face Spaces
- **CPU Basic**: Free (public Spaces)
- **CPU Upgraded**: ~$5/month (faster, more memory)
- **GPU**: $60-300/month (overkill for DistilBERT)

**Recommended**: Free tier for demos, upgraded for production

### API Usage Costs
- **OpenAI GPT-4o-mini**: ~$0.15/1M input tokens, $0.60/1M output
  - ~$0.01-0.03 per explanation
- **Google Fact Check**: 1000 queries/day free, then paid tier

**Estimated API cost**: $5-20/month depending on traffic

---

## Next Steps

After successful deployment:

1. **Custom Domain** (Fly.io):
   ```bash
   flyctl certs create yourdomain.com
   flyctl certs show yourdomain.com
   # Add DNS records as shown
   ```

2. **Authentication**: Add Streamlit auth or OAuth2 for FastAPI

3. **Monitoring**: Set up Sentry, LogTail, or Datadog

4. **CI/CD**: GitHub Actions to auto-deploy on push:
   ```yaml
   # .github/workflows/deploy.yml
   name: Deploy to Fly.io
   on:
     push:
       branches: [main]
   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: superfly/flyctl-actions/setup-flyctl@master
         - run: flyctl deploy --remote-only
           env:
             FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
   ```

5. **Rate Limiting**: Protect API from abuse with middleware

6. **Caching**: Add Redis for fact-check results caching

---

**You're all set! Your FakeScope app is now live and ready to detect fake news at scale.** 🚀

For issues or questions:
- GitHub: https://github.com/enriest/FakeScope/issues
- Fly.io Docs: https://fly.io/docs
- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
