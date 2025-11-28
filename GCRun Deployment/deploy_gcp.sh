#!/bin/bash
set -e

# Load .env file
if [ -f .env ]; then
  export $(cat .env | xargs)
else
  echo "Error: .env file not found. Please copy .env.template to .env and fill in your API keys."
  exit 1
fi

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set in .env file. Please add your Hugging Face token."
  exit 1
fi

# Project ID (extracted from the provided URL: fakescope-207378824568)
PROJECT_ID="nth-hybrid-479110-p2"
REGION="europe-west1"
SERVICE_NAME="fakescope"

echo "Deploying to Google Cloud Run..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Build the image using Cloud Build
echo "Building Docker image..."
# Use cloudbuild.yaml to pass build args
gcloud builds submit --quiet --config cloudbuild.yaml --substitutions=_HF_TOKEN=$HF_TOKEN,_SERVICE_NAME=$SERVICE_NAME . --project $PROJECT_ID

# Deploy to Cloud Run
echo "Deploying service..."
gcloud run deploy $SERVICE_NAME \
  --quiet \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY,GEMINI_API_KEY=$GEMINI_API_KEY,PERPLEXITY_API_KEY=$PERPLEXITY_API_KEY,GOOGLE_FACTCHECK_API_KEY=$GOOGLE_FACTCHECK_API_KEY,FAKESCOPE_LLM_PROVIDER=$FAKESCOPE_LLM_PROVIDER"

echo "Deployment complete!"
