#!/bin/bash
# FakeScope Hugging Face Spaces Deployment Script
# This script automates the deployment of FakeScope to HF Spaces

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FakeScope → Hugging Face Spaces Deploy${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ git not found. Please install git.${NC}"
    exit 1
fi

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}⚠️  huggingface-cli not found. Installing...${NC}"
    pip install huggingface-hub
fi

# Check if user is logged in to HF
echo -e "${YELLOW}[2/7] Checking Hugging Face authentication...${NC}"
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${YELLOW}⚠️  Not logged in to Hugging Face. Please login:${NC}"
    huggingface-cli login
else
    HF_USER=$(huggingface-cli whoami | grep 'username' | awk '{print $2}')
    echo -e "${GREEN}✓ Logged in as: $HF_USER${NC}"
fi

# Get Space name
echo ""
echo -e "${YELLOW}[3/7] Space configuration...${NC}"
read -p "Enter Space name (e.g., fakescope): " SPACE_NAME
SPACE_NAME=${SPACE_NAME:-fakescope}

# Get username
HF_USER=$(huggingface-cli whoami | grep 'username' | awk '{print $2}')
SPACE_REPO="$HF_USER/$SPACE_NAME"

echo -e "${BLUE}Space will be created at: https://huggingface.co/spaces/$SPACE_REPO${NC}"

# Ask if public or private
read -p "Make Space public? (y/n, default: y): " IS_PUBLIC
IS_PUBLIC=${IS_PUBLIC:-y}

# Prepare files
echo ""
echo -e "${YELLOW}[4/7] Preparing deployment files...${NC}"

# Create temp deployment directory
DEPLOY_DIR="hf_space_deploy_temp"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy Spaces files
echo "  → Copying app files..."
cp spaces/app.py "$DEPLOY_DIR/"
cp spaces/requirements.txt "$DEPLOY_DIR/"
cp spaces/README.md "$DEPLOY_DIR/"
cp spaces/.env.example "$DEPLOY_DIR/"

# Copy source modules
echo "  → Copying source modules..."
mkdir -p "$DEPLOY_DIR/src"
cp spaces/src/inference.py "$DEPLOY_DIR/src/"
cp spaces/src/factcheck.py "$DEPLOY_DIR/src/"
cp spaces/src/openai_explain.py "$DEPLOY_DIR/src/"
cp spaces/src/utils.py "$DEPLOY_DIR/src/"
cp spaces/src/__init__.py "$DEPLOY_DIR/src/"

# Create .gitignore
echo "  → Creating .gitignore..."
cat > "$DEPLOY_DIR/.gitignore" << 'EOF'
__pycache__/
*.pyc
.env
.DS_Store
EOF

# Create .gitattributes for Git LFS (if model is included)
cat > "$DEPLOY_DIR/.gitattributes" << 'EOF'
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
EOF

echo -e "${GREEN}✓ Files prepared in $DEPLOY_DIR/${NC}"

# Ask about model
echo ""
echo -e "${YELLOW}[5/7] Model configuration...${NC}"
echo "Choose model loading strategy:"
echo "  1) Load from HF Hub (recommended, smaller Space size)"
echo "  2) Include model in Space (larger, but faster first load)"
read -p "Enter choice (1 or 2, default: 1): " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-1}

if [ "$MODEL_CHOICE" = "2" ]; then
    echo "  → Copying model files (this may take a minute)..."
    cp -r distilbert_fakenews_2stage "$DEPLOY_DIR/"
    echo -e "${GREEN}✓ Model copied to deployment directory${NC}"
else
    echo -e "${BLUE}ℹ Model will be loaded from HF Hub at runtime${NC}"
    echo "  Make sure to set FAKESCOPE_MODEL_DIR in Space secrets!"
fi

# Create Space
echo ""
echo -e "${YELLOW}[6/7] Creating and uploading to Space...${NC}"

cd "$DEPLOY_DIR"

# Initialize git repo
git init
git lfs install

# Create Space on HF
if [ "$IS_PUBLIC" = "y" ]; then
    huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk gradio
else
    huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk gradio --private
fi

# Add remote
git remote add origin "https://huggingface.co/spaces/$SPACE_REPO"

# Add and commit files
git add .
git commit -m "Initial FakeScope deployment"

# Push to Space
echo "  → Pushing to Hugging Face..."
git push -u origin main --force

cd ..

echo -e "${GREEN}✓ Space created and files uploaded!${NC}"

# Configure secrets
echo ""
echo -e "${YELLOW}[7/7] Configuring secrets...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Please set the following secrets in your Space:"
echo ""
echo "1. Go to: https://huggingface.co/spaces/$SPACE_REPO/settings"
echo "2. Click 'Repository secrets'"
echo "3. Add these secrets:"
echo ""
echo -e "${GREEN}Required:${NC}"
echo "  • GOOGLE_FACTCHECK_API_KEY"
echo ""
echo -e "${GREEN}LLM Provider (choose ONE):${NC}"
echo "  • OPENAI_API_KEY (if using OpenAI)"
echo "  • PERPLEXITY_API_KEY (if using Perplexity)"
echo "  • GEMINI_API_KEY (if using Gemini - recommended, free tier)"
echo ""
echo -e "${GREEN}Optional:${NC}"
echo "  • FAKESCOPE_LLM_PROVIDER (openai/perplexity/gemini, default: openai)"
if [ "$MODEL_CHOICE" = "1" ]; then
    echo "  • FAKESCOPE_MODEL_DIR (set to: enri-est/fakescope-distilbert-2stage)"
fi
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Cleanup option
echo ""
read -p "Clean up temporary deployment directory? (y/n, default: y): " CLEANUP
CLEANUP=${CLEANUP:-y}

if [ "$CLEANUP" = "y" ]; then
    cd ..
    rm -rf "$DEPLOY_DIR"
    echo -e "${GREEN}✓ Cleanup complete${NC}"
fi

# Final instructions
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🎉 Deployment Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}Your Space:${NC} https://huggingface.co/spaces/$SPACE_REPO"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Configure secrets (see above)"
echo "  2. Wait for Space to build (~5-10 minutes)"
echo "  3. Test your Space!"
echo ""
echo -e "${BLUE}Monitor build status:${NC}"
echo "  https://huggingface.co/spaces/$SPACE_REPO/logs"
echo ""
