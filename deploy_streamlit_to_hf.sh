#!/bin/bash
# Deploy Streamlit version of FakeScope to Hugging Face Spaces

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FakeScope Streamlit → HF Spaces${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if logged in
echo -e "${YELLOW}[1/5] Checking authentication...${NC}"
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${RED}❌ Not logged in. Please run: huggingface-cli login${NC}"
    exit 1
fi

HF_USER=$(huggingface-cli whoami | grep 'username' | awk '{print $2}')
echo -e "${GREEN}✓ Logged in as: $HF_USER${NC}"

# Get Space name
echo ""
echo -e "${YELLOW}[2/5] Space configuration...${NC}"
read -p "Enter Space name (e.g., fakescope-streamlit): " SPACE_NAME
SPACE_NAME=${SPACE_NAME:-fakescope-streamlit}
SPACE_REPO="$HF_USER/$SPACE_NAME"

echo -e "${BLUE}Space: https://huggingface.co/spaces/$SPACE_REPO${NC}"

# Prepare deployment directory
echo ""
echo -e "${YELLOW}[3/5] Preparing files...${NC}"
DEPLOY_DIR="hf_streamlit_deploy"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy Streamlit-specific files
cp spaces_streamlit/Dockerfile "$DEPLOY_DIR/"
cp spaces_streamlit/README.md "$DEPLOY_DIR/"
cp spaces_streamlit/requirements.txt "$DEPLOY_DIR/"

# Copy source code
echo "  → Copying source code..."
mkdir -p "$DEPLOY_DIR/src"
cp src/app.py "$DEPLOY_DIR/src/"
cp src/inference.py "$DEPLOY_DIR/src/"
cp src/factcheck.py "$DEPLOY_DIR/src/"
cp src/openai_explain.py "$DEPLOY_DIR/src/"
cp src/utils.py "$DEPLOY_DIR/src/"
[ -f src/storage.py ] && cp src/storage.py "$DEPLOY_DIR/src/"
[ -f src/db.py ] && cp src/db.py "$DEPLOY_DIR/src/"
touch "$DEPLOY_DIR/src/__init__.py"

# Copy Streamlit config
mkdir -p "$DEPLOY_DIR/.streamlit"
if [ -f ".streamlit/config.toml" ]; then
    cp .streamlit/config.toml "$DEPLOY_DIR/.streamlit/"
else
    cat > "$DEPLOY_DIR/.streamlit/config.toml" << 'EOF'
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"

[server]
headless = true
enableCORS = false
port = 7860
EOF
fi

# Create .gitignore
cat > "$DEPLOY_DIR/.gitignore" << 'EOF'
__pycache__/
*.pyc
.env
.DS_Store
*.db
EOF

echo -e "${GREEN}✓ Files prepared${NC}"

# Create Space
echo ""
echo -e "${YELLOW}[4/5] Creating Space...${NC}"
cd "$DEPLOY_DIR"
git init
huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk docker || true
git remote add origin "https://huggingface.co/spaces/$SPACE_REPO"

# Commit and push
git add .
git commit -m "Deploy Streamlit version of FakeScope"
echo "  → Pushing to HF..."
git push -u origin main --force

cd ..

echo -e "${GREEN}✓ Space created!${NC}"

# Instructions
echo ""
echo -e "${YELLOW}[5/5] Configuration required${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Configure secrets at:"
echo "https://huggingface.co/spaces/$SPACE_REPO/settings"
echo ""
echo "Required secrets:"
echo "  • FAKESCOPE_MODEL_DIR = enri-est/fakescope-distilbert-2stage"
echo "  • GOOGLE_FACTCHECK_API_KEY = (your key)"
echo "  • GEMINI_API_KEY (or OPENAI_API_KEY or PERPLEXITY_API_KEY)"
echo "  • FAKESCOPE_LLM_PROVIDER = gemini (or openai/perplexity)"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Cleanup
read -p "Clean up deployment directory? (y/n): " CLEANUP
if [ "${CLEANUP:-y}" = "y" ]; then
    cd ..
    rm -rf "$DEPLOY_DIR"
    echo -e "${GREEN}✓ Cleanup complete${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🎉 Streamlit Space Deployed!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Your Space: https://huggingface.co/spaces/$SPACE_REPO"
echo "Build takes ~10-15 minutes (Docker + model download)"
echo ""
