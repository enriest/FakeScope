---
description: Deploy FakeScope to Google Cloud Run
---

# Deploy FakeScope to Google Cloud Run

This workflow guides you through deploying FakeScope to Google Cloud Run (serverless, auto-scaling).

## Prerequisites

1. Install Google Cloud SDK:
```bash
brew install --cask google-cloud-sdk
```

2. Authenticate and set up project:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Deployment Steps

### 1. Enable Required APIs

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

### 2. Create Secrets for API Keys

```bash
# OpenAI API Key
echo -n "your-openai-api-key" | gcloud secrets create OPENAI_API_KEY --data-file=-

# Google Fact Check API Key
echo -n "your-google-factcheck-api-key" | gcloud secrets create GOOGLE_FACTCHECK_API_KEY --data-file=-

# Verify secrets
gcloud secrets list
```

### 3. Set Environment Variables

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1
export SERVICE_NAME=fakescope-app
```

### 4. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create fakescope-repo \
  --repository-format=docker \
  --location=${REGION} \
  --description="FakeScope Docker images"
```

### 5. Build and Push Container Image

// turbo
```bash
gcloud builds submit \
  --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest \
  --timeout=20m
```

### 6. Deploy to Cloud Run

**IMPORTANT**: Replace `YOUR_USERNAME/fakescope-distilbert-2stage` with your actual Hugging Face model repo ID.

```bash
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
```

### 7. Get Service URL

```bash
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)'
```

### 8. Test Deployment

```bash
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)')
curl -I ${SERVICE_URL}
```

## Monitoring

View logs:
```bash
gcloud run services logs tail ${SERVICE_NAME} --region=${REGION}
```

View metrics in Cloud Console:
```
https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics
```

## Update Deployment

To deploy a new version:

```bash
# Rebuild image
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest

# Deploy new version (Cloud Run will automatically route traffic)
gcloud run services update ${SERVICE_NAME} \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/fakescope-repo/fakescope:latest \
  --region ${REGION}
```

## Cost Optimization

Scale to zero when idle (default):
```bash
gcloud run services update ${SERVICE_NAME} --region=${REGION} --min-instances=0
```

Keep 1 instance warm (eliminates cold starts but increases cost):
```bash
gcloud run services update ${SERVICE_NAME} --region=${REGION} --min-instances=1
```

## Troubleshooting

Check logs for errors:
```bash
gcloud run services logs read ${SERVICE_NAME} --region=${REGION} --limit=100
```

Increase timeout if model download is slow:
```bash
gcloud run services update ${SERVICE_NAME} --region=${REGION} --timeout=600s
```

Increase memory if getting OOM errors:
```bash
gcloud run services update ${SERVICE_NAME} --region=${REGION} --memory=4Gi
```

## Cleanup

Delete the service:
```bash
gcloud run services delete ${SERVICE_NAME} --region=${REGION}
```

Delete the container images:
```bash
gcloud artifacts repositories delete fakescope-repo --location=${REGION}
```

Delete secrets:
```bash
gcloud secrets delete OPENAI_API_KEY
gcloud secrets delete GOOGLE_FACTCHECK_API_KEY
```
