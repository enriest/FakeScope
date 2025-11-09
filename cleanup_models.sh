#!/bin/bash
# filepath: cleanup_models.sh
# FakeScope Model Cleanup Script with Optional Backup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_DIR="/Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope"
BACKUP_DIR="${PROJECT_DIR}/model_backups/$(date +%Y%m%d_%H%M%S)"

echo -e "${YELLOW}=== FakeScope Model Cleanup ===${NC}"
echo ""
echo "This script will delete:"
echo "  • Transformer models (distilbert_*)"
echo "  • Training artifacts (results*, mlm_results/)"
echo "  • Baseline models (*.joblib)"
echo "  • Cache files (factcheck_cache.json, cv_*.json)"
echo ""
read -p "Create backup before deletion? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Creating backup at: ${BACKUP_DIR}${NC}"
    mkdir -p "${BACKUP_DIR}"
    
    # Backup transformer models
    for model_dir in distilbert_fakenews distilbert_fakenews_2stage distilbert_news_adapted; do
        if [ -d "${PROJECT_DIR}/${model_dir}" ]; then
            echo "  Backing up ${model_dir}..."
            cp -R "${PROJECT_DIR}/${model_dir}" "${BACKUP_DIR}/"
        fi
    done
    
    # Backup baseline models
    [ -f "${PROJECT_DIR}/best_baseline_model.joblib" ] && cp "${PROJECT_DIR}/best_baseline_model.joblib" "${BACKUP_DIR}/"
    [ -f "${PROJECT_DIR}/tfidf_vectorizer.joblib" ] && cp "${PROJECT_DIR}/tfidf_vectorizer.joblib" "${BACKUP_DIR}/"
    
    echo -e "${GREEN}✅ Backup complete${NC}"
fi

echo ""
read -p "Proceed with deletion? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Aborted${NC}"
    exit 0
fi

cd "${PROJECT_DIR}"

echo -e "${RED}Deleting models...${NC}"

# Delete transformer models
rm -rf distilbert_fakenews/ distilbert_fakenews_2stage/ distilbert_news_adapted/

# Delete training checkpoints
rm -rf results/ results_2stage/ mlm_results/ results_fold*/

# Delete baseline models
rm -f best_baseline_model.joblib tfidf_vectorizer.joblib

# Delete cache/analysis files
rm -f factcheck_cache.json cv_aggregate_results.json model_comparison_summary.json
rm -f cv_vs_2stage_comparison.png

echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run cell #VSC-c7f63fd9 (Class distribution)"
echo "  2. Run cells #VSC-bcf99abf → #VSC-550cd298 (Baseline models)"
echo "  3. Run cell #VSC-c3b8a448 (Stage 1 MLM - ~45-90 min)"
echo "  4. Run cell #VSC-2e6e4fa0 (Stage 2 Fine-tuning - ~15-30 min)"

[ -d "${BACKUP_DIR}" ] && echo "" && echo "Backup location: ${BACKUP_DIR}"