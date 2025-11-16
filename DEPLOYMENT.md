# FakeScope Deployment Guide

This guide explains how to containerize and deploy the FakeScope web app with Docker, using Fly.io for hosting. The app provides:

- A Streamlit UI to analyze an article URL, title, or text
- Two scores: model credibility (0–100) and Google Fact Check aggregate (0–1)
- An OpenAI-generated explanation
- A dashboard tab with recent analyses
 - A language selector for Google Fact Check queries (multilingual lookup)

## Prerequisites

- Docker installed
- Fly.io account and `flyctl` CLI: https://fly.io/docs/hands-on/install-flyctl/
- API keys (set as secrets):
  - `OPENAI_API_KEY`
  - `GOOGLE_FACTCHECK_API_KEY`

## Repository Layout (Deployment-Relevant)

- `Dockerfile` – builds the container
- `src/app.py` – Streamlit UI
- `src/inference.py` – DistilBERT inference
- `src/factcheck.py` – Google Fact Check client
- `src/openai_explain.py` – LLM explanation
- `src/storage.py` – SQLite persistence for dashboard
- `distilbert_fakenews_2stage/` – final classifier (included in image)
- `.dockerignore` – excludes large, nonessential items from build context

Note: Datasets are ignored via `.gitignore` and not required for inference.

## Local Run (Optional)

```bash
# Build image
docker build -t fakescope:latest .

# Run container
docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e GOOGLE_FACTCHECK_API_KEY="$GOOGLE_FACTCHECK_API_KEY" \
  fakescope:latest

# App will be available at http://localhost:8080
# The UI includes a "Fact Check Language" dropdown. This affects ONLY Google Fact Check lookups; the DistilBERT classifier remains English-trained.
```

## Fly.io Deployment

1) Initialize the app

```bash
flyctl launch --no-deploy
```

This will create or update `fly.toml` (already present). You can accept defaults and set the app name if prompted.

2) Create and attach a volume for persistence

```bash
# Create a 1GB volume in your region (e.g., iad)
flyctl volumes create fakescope_data --size 1 --region iad
```

The app mounts this volume at `/data` and stores the SQLite DB at `/data/predictions.db` via `FAKESCOPE_DB_PATH`.

3) Set secrets

```bash
flyctl secrets set OPENAI_API_KEY="$OPENAI_API_KEY"
flyctl secrets set GOOGLE_FACTCHECK_API_KEY="$GOOGLE_FACTCHECK_API_KEY"
```

4) Deploy

```bash
flyctl deploy
```

Once deployed, Fly.io will provide a public URL. Health checks hit `/` on port 8080.

### Optional API (FastAPI)

The container also runs a FastAPI server on port `8001` with:
- `GET /healthz` – health probe
- `POST /predict` – basic JSON prediction API

By default, only the Streamlit port (8080) is exposed. If you want to expose the API, add another `[[services]]` entry in `fly.toml` mapping `internal_port = 8001`.

## Configuration

Environment variables:

- `FAKESCOPE_MODEL_DIR` – path to model folder (default: `./distilbert_fakenews_2stage`)
- `FAKESCOPE_DB_PATH` – SQLite file path (default: `./data/predictions.db`)
- `FAKESCOPE_OPENAI_MODEL` – OpenAI model for explanations (default: `gpt-4o-mini`)

## Notes & Tips

- Container size is large due to PyTorch + DistilBERT. Choose a Fly.io machine size with enough memory (1–2GB recommended).
- If you prefer not to embed the model, host it on a storage bucket or HF Hub and set `FAKESCOPE_MODEL_DIR` to a mounted path, then download on startup.
- Google Fact Check API has daily quotas. The app aggregates the first few results and maps textual ratings to a numeric score.
- If `OPENAI_API_KEY` is not set, the app still runs; the explanation section will show a placeholder.
 - Multilingual: Selecting a non-English language (e.g., `es`, `fr`) queries fact-check sources in that language. For higher-quality classification of non-English text, translate to English before submission or retrain a multilingual model.

### Multilingual Usage
The current model (`distilbert_fakenews_2stage/`) was fine-tuned on English news. For non-English input:
1. Choose the appropriate language in the dropdown so fact-check sources are searched regionally.
2. (Optional) Translate the text to English prior to analysis for better credibility scoring.
3. Compare results: external fact-check sources vs. English-based model output.

### Automatic Translation Option
The UI provides a checkbox (enabled by default) to auto-translate non-English input to English *before* model scoring and LLM explanation using `deep-translator`.

Runtime behavior:
- Fact check queries still use original language text to maximize local source matches.
- If translation fails or is disabled, the original text is used and a status caption indicates this.
- Disable globally by setting `FAKESCOPE_DISABLE_TRANSLATION=1` in the container/VM environment.

Add to Fly secrets or Docker run command:
```bash
flyctl secrets set FAKESCOPE_DISABLE_TRANSLATION=1   # disable translation
```
```bash
docker run -e FAKESCOPE_DISABLE_TRANSLATION=1 ... fakescope:latest
```

Accuracy note: Machine translation may alter nuance (named entities, idioms). For critical misinformation assessments, manually validate the translation or incorporate a higher-quality translation API with confidence scores.

#### Upgrading to True Multilingual Classification
To natively support multiple languages:
- Swap model to `distilbert-base-multilingual-cased` or `xlm-roberta-base`.
- Re-run Stage 1 (MLM) on a combined multilingual news corpus.
- Re-run Stage 2 (classification) with labeled multilingual fake/true samples.
- Store `language_code` in DB for longitudinal performance tracking.

#### Optional Enhancement (Not Implemented Yet)
Add a middleware translation step (e.g., DeepL API) when `language != 'en'` and persist both original and translated text for auditability.

## Troubleshooting

- Model not found: ensure `distilbert_fakenews_2stage/` exists in the Docker build context or set `FAKESCOPE_MODEL_DIR` correctly.
- Port issues: Streamlit binds to `0.0.0.0:8080` in the container and is configured in `fly.toml`.
- Memory errors: Use a larger Fly.io machine or reduce concurrent users.
# FakeScope Deployment Guide

## 📊 Current Status Assessment

### ✅ What's Production-Ready

- **Trained Models**: `distilbert_fakenews_2stage/` achieves 98-99.5% accuracy
- **Data Pipeline**: Robust preprocessing with deduplication and validation
- **Configuration**: Centralized config in `src/config.py`
- **Testing**: Basic test suite with pytest
- **Documentation**: Comprehensive README and code documentation
- **Hardware Optimization**: Apple Silicon MPS support

### ❌ What's Missing for Production

- **REST API**: No FastAPI/Flask endpoint (estimated: 4-6 hours)
- **Dockerfiles**: No containerization (estimated: 2-3 hours)
- **CI/CD**: No automated deployment pipeline (estimated: 3-4 hours)
- **Monitoring**: No logging/metrics collection (estimated: 4-6 hours)
- **API Tests**: No integration tests for API (estimated: 3-4 hours)
- **Rate Limiting**: No request throttling (estimated: 2-3 hours)

**Total Gap**: ~18-26 hours of development work

---

## 🚀 Deployment Strategy

### Phase 1: Local API (Quick Start - 4-6 hours)

#### Step 1.1: Create FastAPI Server

Create `src/api.py`:

```python
"""
FastAPI server for FakeScope inference.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
from typing import Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FakeScope API",
    description="Fake news detection API using DistilBERT",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (loaded on startup)
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    """Load model on server startup."""
    global model, tokenizer
    logger.info("Loading model...")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            './distilbert_fakenews_2stage'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            './distilbert_fakenews_2stage'
        )
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000, 
                     description="News article text to analyze")
    use_baseline: bool = Field(default=False, 
                              description="Use traditional ML fallback")
    return_probabilities: bool = Field(default=True,
                                      description="Return class probabilities")

class PredictionResponse(BaseModel):
    credibility_score: float = Field(..., ge=0, le=100)
    label: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Optional[dict] = None
    processing_time_ms: float

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "model": "distilbert-fakenews-2stage",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credibility of news article.
    
    Args:
        request: PredictionRequest with text to analyze
        
    Returns:
        PredictionResponse with credibility score and label
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Extract results
        fake_prob = probs[0].item()
        true_prob = probs[1].item()
        credibility_score = true_prob * 100
        label = "TRUE" if true_prob > 0.5 else "FAKE"
        confidence = max(fake_prob, true_prob)
        
        processing_time = (time.time() - start_time) * 1000
        
        response = PredictionResponse(
            credibility_score=round(credibility_score, 2),
            label=label,
            confidence=round(confidence, 4),
            processing_time_ms=round(processing_time, 2)
        )
        
        if request.return_probabilities:
            response.probabilities = {
                "FAKE": round(fake_prob, 4),
                "TRUE": round(true_prob, 4)
            }
        
        logger.info(f"Prediction: {label} ({credibility_score:.1f}%) in {processing_time:.1f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(texts: list[str]):
    """Batch prediction endpoint (future feature)."""
    raise HTTPException(status_code=501, detail="Batch prediction not yet implemented")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Step 1.2: Install API Dependencies

Add to `requirements.txt`:
```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
```

Install:
```bash
pip install fastapi uvicorn[standard] pydantic
```

#### Step 1.3: Run Local API

```bash
# From project root
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

#### Step 1.4: Test API

```bash
# Health check
curl http://localhost:8000/health

# Sample prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Scientists announce breakthrough in renewable energy technology that could revolutionize power generation",
    "return_probabilities": true
  }'

# Expected response
{
  "credibility_score": 78.45,
  "label": "TRUE",
  "confidence": 0.7845,
  "probabilities": {
    "FAKE": 0.2155,
    "TRUE": 0.7845
  },
  "processing_time_ms": 45.2
}
```

---

### Phase 2: Dockerization (2-3 hours)

#### Step 2.1: Create Dockerfile

Create `Dockerfile`:

```dockerfile
# FakeScope Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY distilbert_fakenews_2stage/ ./distilbert_fakenews_2stage/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 2.2: Create .dockerignore

Create `.dockerignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
.venv/
venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data
datasets/
mlruns/
mlm_results/
results_*/

# Git
.git/
.gitignore

# Docs
*.md
Documents/

# Tests
tests/
htmlcov/
.coverage
```

#### Step 2.3: Build and Test Docker Image

```bash
# Build image
docker build -t fakescope-api:v1 .

# Check image size
docker images fakescope-api:v1

# Run container
docker run -d \
  --name fakescope \
  -p 8000:8000 \
  fakescope-api:v1

# Test API
curl http://localhost:8000/health

# View logs
docker logs fakescope

# Stop container
docker stop fakescope
docker rm fakescope
```

---

### Phase 3: Cloud Deployment (3-6 hours)

#### Option 3A: Google Cloud Run (Recommended - Easiest)

```bash
# 1. Install Google Cloud SDK
# Follow: https://cloud.google.com/sdk/docs/install

# 2. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fakescope-api

# 4. Deploy to Cloud Run
gcloud run deploy fakescope-api \
  --image gcr.io/YOUR_PROJECT_ID/fakescope-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 60s \
  --max-instances 10

# 5. Get URL
gcloud run services describe fakescope-api --region us-central1 --format 'value(status.url)'
```

**Costs**: ~$0.10-0.50/day for light usage (includes 2M free requests/month)

#### Option 3B: AWS EC2 + Docker

```bash
# 1. Launch EC2 instance (t3.medium or t3.large)
# 2. SSH into instance
ssh -i your-key.pem ec2-user@YOUR_EC2_IP

# 3. Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# 4. Pull and run image
docker pull YOUR_DOCKERHUB_USERNAME/fakescope-api:v1
docker run -d -p 80:8000 --restart always YOUR_DOCKERHUB_USERNAME/fakescope-api:v1

# 5. Configure security group to allow port 80
```

**Costs**: ~$30-60/month (t3.medium = ~$0.04/hr)

#### Option 3C: HuggingFace Spaces (Free Tier Available)

Create `app.py` in project root:

```python
"""
Gradio interface for HuggingFace Spaces deployment.
"""
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained('./distilbert_fakenews_2stage')
tokenizer = AutoTokenizer.from_pretrained('./distilbert_fakenews_2stage')

def predict_credibility(text):
    """Predict credibility of news article."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    
    credibility = probs[1].item() * 100
    label = "TRUE" if probs[1] > 0.5 else "FAKE"
    confidence = max(probs).item() * 100
    
    return f"""
    🎯 **Classification**: {label}
    📊 **Credibility Score**: {credibility:.1f}/100
    ✅ **Confidence**: {confidence:.1f}%
    
    📈 **Probabilities**:
    - FAKE: {probs[0].item()*100:.1f}%
    - TRUE: {probs[1].item()*100:.1f}%
    """

# Create Gradio interface
demo = gr.Interface(
    fn=predict_credibility,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste news article text here...",
        label="News Article"
    ),
    outputs=gr.Markdown(label="Analysis Results"),
    title="🔍 FakeScope: Fake News Detector",
    description="AI-powered fake news detection using DistilBERT (98-99.5% accuracy)",
    examples=[
        ["Scientists announce breakthrough in renewable energy that could revolutionize power generation."],
        ["BREAKING: Celebrity reveals shocking secret about government conspiracy."],
    ]
)

if __name__ == "__main__":
    demo.launch()
```

Deploy:
```bash
# 1. Create space at huggingface.co/spaces
# 2. Clone space
git clone https://huggingface.co/spaces/YOUR_USERNAME/fakescope
cd fakescope

# 3. Copy files
cp app.py fakescope/
cp -r distilbert_fakenews_2stage/ fakescope/
cp requirements.txt fakescope/

# 4. Push to HuggingFace
git add .
git commit -m "Initial deployment"
git push
```

**Costs**: Free (with rate limits)

---

### Phase 4: Monitoring & Scaling (4-6 hours)

#### Step 4.1: Add Logging

Update `src/api.py`:

```python
import logging
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/api_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
```

#### Step 4.2: Add Prometheus Metrics

```python
from prometheus_fastapi_instrumentator import Instrumentator

# Add to FastAPI app
Instrumentator().instrument(app).expose(app)
```

#### Step 4.3: Database for Request Logging

```python
# Use SQLite for lightweight logging
import sqlite3

def log_prediction(text_hash, prediction, confidence):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (timestamp, text_hash, prediction, confidence)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now(), text_hash, prediction, confidence))
    conn.commit()
    conn.close()
```

---

## 🎯 Quick Deployment Path (Recommended)

**For immediate deployment with minimal effort**:

1. **Implement REST API** (4-6 hours) → Follow Phase 1
2. **Deploy to HuggingFace Spaces** (1 hour) → Follow Phase 3C
3. **Share public URL** → Done!

**Total Time**: ~5-7 hours

**Pros**:
- Free hosting
- No DevOps complexity
- Automatic HTTPS
- Built-in rate limiting

**Cons**:
- Limited to 2GB RAM
- Slower cold starts
- Public visibility

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Verify models exist (`distilbert_fakenews_2stage/`)
- [ ] Test predictions locally in notebook
- [ ] Create `src/api.py` with FastAPI endpoints
- [ ] Add FastAPI dependencies to `requirements.txt`
- [ ] Test API locally (`uvicorn src.api:app`)

### Containerization
- [ ] Create `Dockerfile`
- [ ] Create `.dockerignore`
- [ ] Build Docker image
- [ ] Test Docker container locally
- [ ] Push image to registry (Docker Hub / GCR / ECR)

### Cloud Deployment
- [ ] Choose platform (Cloud Run / EC2 / HuggingFace)
- [ ] Configure environment variables (API keys if needed)
- [ ] Deploy container
- [ ] Test deployed API endpoint
- [ ] Configure custom domain (optional)

### Post-Deployment
- [ ] Set up monitoring (logs, metrics)
- [ ] Configure alerts for errors
- [ ] Document API endpoints (Swagger/OpenAPI)
- [ ] Test with production load
- [ ] Create backup/rollback plan

---

## 🔒 Security Considerations

### API Security
```python
# Add API key authentication
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ... rest of prediction logic
```

### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: PredictionRequest):
    # ... prediction logic
```

---

## 💰 Cost Estimates

| Platform | Setup Time | Monthly Cost | Pros | Cons |
|----------|-----------|--------------|------|------|
| **HuggingFace Spaces** | 1 hour | Free | Easy, no DevOps | Limited resources |
| **Google Cloud Run** | 2 hours | $5-20 | Auto-scaling, pay-per-use | Requires GCP account |
| **AWS EC2** | 3 hours | $30-60 | Full control | Manual scaling |
| **Digital Ocean** | 3 hours | $12-24 | Simple, affordable | Manual maintenance |

**Recommendation**: Start with **HuggingFace Spaces** for validation, migrate to **Cloud Run** for production scale.

---

## 📞 Support

For deployment issues, check:
- **Model Loading**: Ensure `distilbert_fakenews_2stage/` exists
- **Memory Errors**: Increase container memory (2GB minimum)
- **Slow Inference**: Use GPU instance or optimize batch size
- **API Errors**: Check logs with `docker logs` or cloud platform logs

---

**Last Updated**: November 15, 2025
