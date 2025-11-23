# 🚀 Google Cloud Platform Deployment Guide for FakeScope

This guide provides step-by-step instructions for deploying FakeScope to Google Cloud Platform (GCP) using various deployment options.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Options Overview](#deployment-options-overview)
- [Option 1: Cloud Run (Recommended)](#option-1-cloud-run-recommended)
- [Option 2: Google Kubernetes Engine (GKE)](#option-2-google-kubernetes-engine-gke)
- [Option 3: Compute Engine VM](#option-3-compute-engine-vm)
- [Environment Variables & Secrets](#environment-variables--secrets)
- [Monitoring & Logging](#monitoring--logging)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### 1. Google Cloud Account Setup

```bash
# Install Google Cloud SDK
# macOS (using Homebrew)
brew install --cask google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Windows: Download from https://cloud.google.com/sdk/docs/install
```

### 2. Initialize gcloud CLI

```bash
# Authenticate with Google Cloud
gcloud auth login

# Create a new project or use existing one
gcloud projects create fakescope-prod --name="FakeScope Production"

# Set the project as default
gcloud config set project fakescope-prod

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  secretmanager.googleapis.com
```

### 3. Set Up API Keys as Secrets

```bash
# Create secrets for API keys
echo -n "your-openai-api-key" | gcloud secrets create OPENAI_API_KEY --data-file=-
echo -n "your-google-factcheck-api-key" | gcloud secrets create GOOGLE_FACTCHECK_API_KEY --data-file=-

# Optional: Perplexity or Gemini API keys
echo -n "your-perplexity-api-key" | gcloud secrets create PERPLEXITY_API_KEY --data-file=-
echo -n "your-gemini-api-key" | gcloud secrets create GEMINI_API_KEY --data-file=-

# Verify secrets were created
gcloud secrets list
```

### 4. Prepare Your Model

You have two options for the model:

**Option A: Include in Docker image (simpler but larger)**
- Model will be bundled in the container (~1.5GB image)

**Option B: Load from Hugging Face Hub (recommended)**
```bash
# Upload your model to Hugging Face (if not already done)
pip install huggingface_hub
export HF_TOKEN=hf_your_write_token

python scripts/upload_model_hf.py \
  --repo-id YOUR_USERNAME/fakescope-distilbert-2stage \
  --model-dir models/distilbert_fakenews_2stage \
  --private
```

---

## 🎯 Deployment Options Overview

| Option | Complexity | Cost (est.) | Scaling | Best For |
|--------|-----------|-------------|---------|----------|
| **Cloud Run** | ⭐ Easy | $5-20/month | Auto | Production, serverless |
| **GKE** | ⭐⭐⭐ Advanced | $70+/month | Manual/Auto | Enterprise, microservices |
| **Compute Engine** | ⭐⭐ Medium | $30-60/month | Manual | Full control, custom setup |

**Recommendation**: Start with **Cloud Run** for simplicity and cost-effectiveness.

---

## Option 1: Cloud Run (Recommended)

Cloud Run is a fully managed serverless platform that automatically scales your containerized application.

### Step 1: Create Dockerfile for Cloud Run

Your existing `Dockerfile` should work, but let's create an optimized version:

```bash
# Create Dockerfile.cloudrun (optional optimization)
cat > Dockerfile.cloudrun << 'EOF'
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy application code
COPY src ./src

# Create data directory for SQLite
RUN mkdir -p /app/data

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Run Streamlit (Cloud Run will set PORT automatically)
CMD streamlit run src/app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.serverAddress=0.0.0.0 \
    --browser.gatherUsageStats=false
EOF
```

### Step 2: Build and Push Container to Artifact Registry

```bash
# Set your project ID
export PROJECT_ID=fakescope-prod
export REGION=us-central1
export SERVICE_NAME=fakescope-app

# Create Artifact Registry repository (one-time setup)
gcloud artifacts repositories create fakescope-repo \
  --repository-format=docker \
  --location=${REGION} \
  --description="FakeScope Docker images"

# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push the image using Cloud Build (recommended - faster)
gcloud builds submit \
  --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest \
  --timeout=20m

# Alternative: Build locally and push (slower)
# docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest .
# docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest
```

### Step 3: Deploy to Cloud Run

```bash
# Deploy with secrets and environment variables
gcloud run deploy ${SERVICE_NAME} \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest,GOOGLE_FACTCHECK_API_KEY=GOOGLE_FACTCHECK_API_KEY:latest \
  --set-env-vars FAKESCOPE_MODEL_DIR=YOUR_USERNAME/fakescope-distilbert-2stage,FAKESCOPE_LLM_PROVIDER=openai,FAKESCOPE_DB_PATH=/app/data/predictions.db

# Get the service URL
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)'
```

### Step 4: Configure Custom Domain (Optional)

```bash
# Map a custom domain to your Cloud Run service
gcloud run domain-mappings create \
  --service ${SERVICE_NAME} \
  --domain fakescope.yourdomain.com \
  --region ${REGION}

# Follow the instructions to update your DNS records
```

### Step 5: Set Up Cloud Storage for Persistent Data (Optional)

Cloud Run is stateless, so SQLite data will be lost on restart. For persistence:

```bash
# Create a Cloud Storage bucket for database backups
gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://fakescope-data-backup/

# Add a backup script to your application or use Cloud Scheduler
# to periodically backup /app/data/predictions.db to Cloud Storage
```

**Better Option**: Use Cloud SQL for PostgreSQL instead of SQLite for production:

```bash
# Create Cloud SQL instance
gcloud sql instances create fakescope-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=${REGION}

# Create database
gcloud sql databases create fakescope --instance=fakescope-db

# Update your app to use PostgreSQL instead of SQLite
# Modify src/storage.py to use SQLAlchemy with PostgreSQL
```

---

## Option 2: Google Kubernetes Engine (GKE)

For advanced users who need more control and orchestration capabilities.

### Step 1: Create GKE Cluster

```bash
# Create a GKE cluster (autopilot mode - easier management)
gcloud container clusters create-auto fakescope-cluster \
  --region=${REGION} \
  --release-channel=regular

# Get cluster credentials
gcloud container clusters get-credentials fakescope-cluster --region=${REGION}

# Verify connection
kubectl cluster-info
```

### Step 2: Create Kubernetes Secrets

```bash
# Create secrets from GCP Secret Manager
kubectl create secret generic fakescope-secrets \
  --from-literal=OPENAI_API_KEY=$(gcloud secrets versions access latest --secret=OPENAI_API_KEY) \
  --from-literal=GOOGLE_FACTCHECK_API_KEY=$(gcloud secrets versions access latest --secret=GOOGLE_FACTCHECK_API_KEY)
```

### Step 3: Create Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fakescope-deployment
  labels:
    app: fakescope
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fakescope
  template:
    metadata:
      labels:
        app: fakescope
    spec:
      containers:
      - name: fakescope
        image: us-central1-docker.pkg.dev/fakescope-prod/fakescope-repo/fakescope:latest
        ports:
        - containerPort: 8080
          name: streamlit
        - containerPort: 8001
          name: api
        env:
        - name: FAKESCOPE_MODEL_DIR
          value: "YOUR_USERNAME/fakescope-distilbert-2stage"
        - name: FAKESCOPE_LLM_PROVIDER
          value: "openai"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: fakescope-secrets
              key: OPENAI_API_KEY
        - name: GOOGLE_FACTCHECK_API_KEY
          valueFrom:
            secretKeyRef:
              name: fakescope-secrets
              key: GOOGLE_FACTCHECK_API_KEY
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: fakescope-service
spec:
  type: LoadBalancer
  selector:
    app: fakescope
  ports:
  - name: streamlit
    port: 80
    targetPort: 8080
  - name: api
    port: 8001
    targetPort: 8001
```

### Step 4: Deploy to GKE

```bash
# Apply the deployment
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# Get external IP (may take a few minutes)
kubectl get service fakescope-service --watch

# Access your app at http://<EXTERNAL-IP>
```

### Step 5: Set Up Horizontal Pod Autoscaling

```bash
# Create HPA based on CPU usage
kubectl autoscale deployment fakescope-deployment \
  --cpu-percent=70 \
  --min=2 \
  --max=10

# Check HPA status
kubectl get hpa
```

---

## Option 3: Compute Engine VM

For users who want full control over the infrastructure.

### Step 1: Create VM Instance

```bash
# Create a VM with container-optimized OS
gcloud compute instances create-with-container fakescope-vm \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --boot-disk-size=50GB \
  --container-image=${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest \
  --container-restart-policy=always \
  --container-env=FAKESCOPE_MODEL_DIR=YOUR_USERNAME/fakescope-distilbert-2stage,FAKESCOPE_LLM_PROVIDER=openai \
  --container-env-file=<(gcloud secrets versions access latest --secret=OPENAI_API_KEY | sed 's/^/OPENAI_API_KEY=/') \
  --tags=http-server,https-server

# Create firewall rule to allow HTTP traffic
gcloud compute firewall-rules create allow-fakescope \
  --allow tcp:8080 \
  --target-tags http-server \
  --description="Allow traffic to FakeScope on port 8080"

# Get external IP
gcloud compute instances describe fakescope-vm \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

### Step 2: SSH into VM and Configure (Alternative Manual Setup)

```bash
# SSH into the VM
gcloud compute ssh fakescope-vm --zone=us-central1-a

# Inside the VM, install Docker (if not using container-optimized OS)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker

# Pull and run the container
sudo docker pull ${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest

sudo docker run -d \
  --name fakescope \
  --restart always \
  -p 8080:8080 \
  -e OPENAI_API_KEY="your-key" \
  -e GOOGLE_FACTCHECK_API_KEY="your-key" \
  -e FAKESCOPE_MODEL_DIR="YOUR_USERNAME/fakescope-distilbert-2stage" \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest

# Check logs
sudo docker logs -f fakescope
```

### Step 3: Set Up Automatic Backups

```bash
# Create a snapshot schedule for the VM disk
gcloud compute resource-policies create snapshot-schedule fakescope-daily-backup \
  --region=${REGION} \
  --max-retention-days=7 \
  --on-source-disk-delete=keep-auto-snapshots \
  --daily-schedule \
  --start-time=02:00

# Attach the schedule to the VM disk
gcloud compute disks add-resource-policies fakescope-vm \
  --resource-policies=fakescope-daily-backup \
  --zone=us-central1-a
```

---

## 🔐 Environment Variables & Secrets

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM explanations | `sk-...` |
| `GOOGLE_FACTCHECK_API_KEY` | Google Fact Check API key | `AIza...` |
| `FAKESCOPE_MODEL_DIR` | Model directory or HF repo ID | `username/fakescope-distilbert-2stage` |
| `FAKESCOPE_LLM_PROVIDER` | LLM provider (openai/perplexity/gemini) | `openai` |
| `FAKESCOPE_DB_PATH` | SQLite database path | `/app/data/predictions.db` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FAKESCOPE_OPENAI_MODEL` | OpenAI model to use | `gpt-4o-mini` |
| `FAKESCOPE_DISABLE_TRANSLATION` | Disable auto-translation | `0` |
| `PORT` | Server port (Cloud Run sets this) | `8080` |

### Managing Secrets in GCP

```bash
# Update a secret
echo -n "new-api-key" | gcloud secrets versions add OPENAI_API_KEY --data-file=-

# Grant Cloud Run access to secrets
gcloud secrets add-iam-policy-binding OPENAI_API_KEY \
  --member=serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor

# List all secrets
gcloud secrets list

# View secret metadata (not the actual value)
gcloud secrets describe OPENAI_API_KEY
```

---

## 📊 Monitoring & Logging

### Cloud Run Monitoring

```bash
# View logs
gcloud run services logs read ${SERVICE_NAME} \
  --region=${REGION} \
  --limit=50

# Stream logs in real-time
gcloud run services logs tail ${SERVICE_NAME} --region=${REGION}

# View metrics in Cloud Console
# Navigate to: Cloud Run → fakescope-app → Metrics
```

### Set Up Alerts

```bash
# Create an alert policy for high error rate
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="FakeScope High Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s
```

### Application Performance Monitoring

Add Cloud Trace and Cloud Profiler to your application:

```python
# Add to requirements.txt
google-cloud-trace>=1.11.0
google-cloud-profiler>=4.0.0

# Add to src/app.py
import googlecloudprofiler
from google.cloud import trace_v1

# Enable profiler
try:
    googlecloudprofiler.start(
        service='fakescope',
        service_version='1.0.0',
        verbose=3
    )
except Exception as e:
    print(f"Failed to start profiler: {e}")
```

---

## 💰 Cost Optimization

### Cloud Run Cost Optimization

```bash
# Set minimum instances to 0 (scale to zero when idle)
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --min-instances=0

# Use smaller CPU allocation (1 vCPU instead of 2)
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --cpu=1 \
  --memory=2Gi

# Set request timeout to avoid long-running requests
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --timeout=60s

# Set concurrency (requests per container instance)
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --concurrency=80
```

### Estimated Monthly Costs

**Cloud Run (Recommended)**
- Free tier: 2M requests, 360,000 GB-seconds, 180,000 vCPU-seconds
- Light usage (1000 requests/day): **$5-10/month**
- Medium usage (10,000 requests/day): **$15-30/month**
- Heavy usage (100,000 requests/day): **$50-100/month**

**GKE Autopilot**
- Minimum: **$70-100/month** (cluster management + 2 pods)
- Medium: **$150-250/month** (with autoscaling)

**Compute Engine**
- e2-standard-2 (2 vCPU, 8GB RAM): **$50-60/month**
- e2-medium (1 vCPU, 4GB RAM): **$25-30/month**

### Cost Monitoring

```bash
# Set up budget alerts
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="FakeScope Monthly Budget" \
  --budget-amount=50USD \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Container Fails to Start

```bash
# Check Cloud Run logs
gcloud run services logs read ${SERVICE_NAME} --region=${REGION} --limit=100

# Common causes:
# - Missing environment variables
# - Model download timeout (increase timeout or use smaller model)
# - Port mismatch (ensure app listens on $PORT)
```

#### 2. Model Download Timeout

```bash
# Increase Cloud Run timeout
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --timeout=600s

# Or include model in Docker image (increases image size)
# Remove line 28-30 from Dockerfile and add:
# COPY models/distilbert_fakenews_2stage ./models/distilbert_fakenews_2stage
```

#### 3. Out of Memory Errors

```bash
# Increase memory allocation
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --memory=4Gi

# Check memory usage in logs
gcloud run services logs read ${SERVICE_NAME} --region=${REGION} | grep -i "memory"
```

#### 4. Slow Cold Starts

```bash
# Set minimum instances to keep containers warm
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --min-instances=1

# Note: This will increase costs but eliminate cold starts
```

#### 5. API Key Errors

```bash
# Verify secrets exist
gcloud secrets list

# Check secret access permissions
gcloud secrets get-iam-policy OPENAI_API_KEY

# Grant access to Cloud Run service account
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding OPENAI_API_KEY \
  --member=serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

### Debug Mode

```bash
# Deploy with debug environment variable
gcloud run services update ${SERVICE_NAME} \
  --region=${REGION} \
  --set-env-vars DEBUG=true,LOG_LEVEL=DEBUG

# SSH into a Cloud Run container (for debugging)
gcloud run services proxy ${SERVICE_NAME} --region=${REGION}
```

### Health Check Endpoints

Test your deployment:

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')

# Test Streamlit health endpoint
curl ${SERVICE_URL}/_stcore/health

# Test if app is responding
curl -I ${SERVICE_URL}
```

---

## 🎯 Quick Deployment Checklist

- [ ] Install and configure `gcloud` CLI
- [ ] Create GCP project and enable APIs
- [ ] Upload API keys to Secret Manager
- [ ] Upload model to Hugging Face Hub (optional but recommended)
- [ ] Build and push Docker image to Artifact Registry
- [ ] Deploy to Cloud Run with secrets and environment variables
- [ ] Test deployment with sample request
- [ ] Configure custom domain (optional)
- [ ] Set up monitoring and alerts
- [ ] Configure budget alerts
- [ ] Document service URL and credentials

---

## 📚 Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)

---

## 🆘 Support

For deployment issues:
1. Check Cloud Run logs: `gcloud run services logs read ${SERVICE_NAME}`
2. Verify environment variables and secrets
3. Test Docker image locally first
4. Check GCP quotas and billing

**Need help?** Open an issue on the [FakeScope GitHub repository](https://github.com/enriest/FakeScope/issues)

---

**Last Updated**: November 23, 2025
