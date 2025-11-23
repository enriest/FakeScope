# FakeScope AI Coding Agent Instructions

## Project Overview
**FakeScope** is a fake news detection system combining ML/NLP models, transformer architectures (DistilBERT), and external fact-checking APIs to provide credibility scores (0-100) with automatic explanations. The project has **dual modes**: research notebooks for experimentation and production `src/` module for deployment.

## Architecture: Research vs. Production

### Research Mode (Notebooks)
- **Primary**: `notebooks/Development.ipynb` (2407 lines) - Full training pipeline
- **Combined**: `notebooks/Project.ipynb` (5319 lines) - Merged pipeline for new work
- **Use when**: Training models, exploratory analysis, parameter tuning

### Production Mode (src/)
- **Web UI**: `src/app.py` - Streamlit app (800 lines, 3 LLM providers)
- **API**: `src/api.py` - FastAPI endpoint (optional, port 8001)
- **Inference**: `src/inference.py` - Model loading with HF Hub fallback
- **Fact-checking**: `src/factcheck.py` - Google API with retry logic
- **LLM**: `src/openai_explain.py` - Multi-provider (OpenAI/Gemini/Perplexity)
- **Storage**: `src/storage.py` - SQLite predictions database
- **Translation**: `src/translate.py` - deep-translator integration
- **Config**: `src/config.py` - Dataclass-based configuration
- **Use when**: Running inference, deploying, building features

## Key Architecture Decisions

### 2-Stage Training Pipeline (Critical Pattern)
The project uses **domain adaptation** before classification to improve performance on news text:

1. **Stage 1**: Masked Language Modeling (MLM) on unlabeled news corpus → saves to `distilbert_news_adapted/`
2. **Stage 2**: Fine-tune adapted model on labeled fake/true news → saves to `distilbert_fakenews_2stage/`

**Why**: Adapts base transformer vocabulary to news domain before fake/true classification. Typically yields +1-3% accuracy vs. standard fine-tuning.

**When to rerun**: 
- MLM stage (8 epochs): Only when adding significant new unlabeled data (e.g., 10k+ articles)
- Classification stage: Every time labels or class balance changes

### Model Storage Convention
- `distilbert_news_adapted/` → Stage 1 MLM output (safetensors + tokenizer)
- `distilbert_fakenews_2stage/` → Stage 2 classifier (final model)
- `best_baseline_model.joblib` + `tfidf_vectorizer.joblib` → Fallback traditional models (LogReg/RF)

Load models in this order of preference: `fakenews_2stage` → `news_adapted` → baseline.

### Data Preprocessing Pattern (Project-Specific)
```python
# Critical: Custom stopwords include publisher names & boilerplate
custom_stopwords = {'reuters', 'ap', 'reporting', 'editing', 'featured', 'image', 'https', 'twitter', 'com', 'getty', 'monday', ...}

# Deduplication: Hash-based to prevent train/test leakage
df_news['content_hash'] = df_news['clean_text'].apply(lambda s: hashlib.md5(s.encode()).hexdigest())

# Train/test split: Group-aware by content_hash to prevent duplicate leakage
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(df_news, df_news['class'], groups=df_news['content_hash']))
```

**Why publisher names matter**: Models can shortcut by memorizing source names rather than content. Always check `CountVectorizer` top tokens for artifacts.

## Hardware-Specific Configuration (Apple Silicon M4)

```python
# TrainingArguments for MacBook Air M4
training_args = TrainingArguments(
    use_mps_device=True,          # Apple Silicon GPU acceleration
    fp16=False,                   # MPS doesn't support fp16, must use fp32
    per_device_train_batch_size=16, # Optimal for M4 memory
    num_train_epochs=3,            # Production: 8 for MLM, 3 for classification
)
```

**Troubleshooting**: If you see `PYTORCH_ENABLE_MPS_FALLBACK`, check for unsupported ops (rare in transformers 4.44+).

## Critical Developer Workflows

### Local Development
```bash
# 1. Environment setup (zsh on macOS)
source .venv/bin/activate
export OPENAI_API_KEY="sk-..." GOOGLE_FACTCHECK_API_KEY="..." 

# 2. Run Streamlit UI
streamlit run src/app.py  # Opens http://localhost:8501

# 3. Run with specific LLM provider
FAKESCOPE_LLM_PROVIDER=gemini streamlit run src/app.py  # Options: openai/gemini/perplexity

# 4. Test APIs
python test_apis.py  # Validates Google/OpenAI/Gemini/Perplexity
```

### Running Tests
```bash
# Quick test suite
pytest tests/ -v

# With coverage (fails below 10%)
pytest tests/ --cov=src --cov-report=html

# View coverage
open htmlcov/index.html  # macOS
```

### Model Loading Pattern
```python
# src/inference.py uses LRU cache and supports 3 modes:

# 1. Local directory (default)
FAKESCOPE_MODEL_DIR=./models/distilbert_fakenews_2stage python -c "from src.inference import credibility_score; print(credibility_score('test'))"

# 2. HuggingFace Hub repo ID
FAKESCOPE_MODEL_DIR=enriest/fakescope-distilbert-2stage python ...

# 3. Full HF URL (auto-normalized to repo ID)
FAKESCOPE_MODEL_DIR=https://huggingface.co/enriest/fakescope-distilbert-2stage python ...

# Private repos need HF_TOKEN or HUGGINGFACE_TOKEN env var
```

### Docker Deployment
```bash
# Build (excludes model from image - downloads from HF Hub)
docker build -t fakescope:latest .

# Run locally
docker run -p 8080:8080 \
  -e FAKESCOPE_MODEL_DIR=enriest/fakescope-distilbert-2stage \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e GOOGLE_FACTCHECK_API_KEY=$GOOGLE_FACTCHECK_API_KEY \
  fakescope:latest

# Deploy to Fly.io (see DEPLOYMENT.md)
flyctl launch --no-deploy
flyctl volumes create fakescope_data --size 1
flyctl secrets set OPENAI_API_KEY="..." GOOGLE_FACTCHECK_API_KEY="..."
flyctl deploy
```

### Training Pipeline (Research)
The project uses **domain adaptation** before classification:

1. **Stage 1 (MLM)**: Masked Language Modeling on unlabeled news corpus
   - Input: 45K+ unlabeled articles from `datasets/input/`
   - Output: `models/distilbert_news_adapted/` (268MB)
   - Duration: 8 epochs (~1.5hr on M4 Mac)
   - **When to rerun**: Only when adding 10K+ new unlabeled data

2. **Stage 2 (Classification)**: Fine-tune adapted model on labeled data
   - Input: Labeled fake/true from `News.csv` + `New Task.csv`
   - Output: `models/distilbert_fakenews_2stage/` (268MB) ← **production model**
   - Duration: 3 epochs (~30min on M4)
   - **When to rerun**: Every time labels or class balance changes

**Why 2-stage**: Adapts base DistilBERT vocabulary to news domain → +1-3% accuracy boost (98-99.5% vs. 97-99%).

### Hardware-Specific Configuration (Apple Silicon M4)
```python
training_args = TrainingArguments(
    use_mps_device=True,          # Apple Silicon GPU (3-4x faster than CPU)
    fp16=False,                   # MPS requires fp32 (no fp16 support)
    per_device_train_batch_size=16, # Optimal for M4 memory
    num_train_epochs=3,            # 8 for MLM, 3 for classification
)
```

## LLM Provider System (Production)

### Multi-Provider Architecture
FakeScope supports 3 LLM providers via `src/openai_explain.py`:

```python
# Switch provider via environment variable
FAKESCOPE_LLM_PROVIDER=openai   # Default: gpt-4o-mini
FAKESCOPE_LLM_PROVIDER=gemini   # Google: gemini-2.5-flash (FREE 1500/day)
FAKESCOPE_LLM_PROVIDER=perplexity # Perplexity: sonar-pro (real-time web search)

# Override model
FAKESCOPE_OPENAI_MODEL=gpt-4o
FAKESCOPE_GEMINI_MODEL=gemini-2.5-pro
FAKESCOPE_PERPLEXITY_MODEL=sonar-reasoning
```

### Gemini REST API Pattern (Critical)
```python
# Uses direct REST API to avoid SDK v1beta issues
url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
payload = {
    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1000}
}
resp = requests.post(url, params={"key": api_key}, json=payload, timeout=20)
```

**Why REST over SDK**: Gemini SDK has stability issues with v1beta models. REST API works reliably with model fallback (`gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-2.5-flash-lite`).

### Prompt Structure (3 Types)
1. **Teacher-Student Review** (temp=0.2): Fact-checking instructions comparing model vs. external sources
2. **Explain Not-Fake** (temp=0.3): Layman explanation for credible claims
3. **Model Understanding** (temp=0.4): Meta-analysis of model behavior

## External APIs

### Google Fact Check (src/factcheck.py)
```python
# Returns normalized scores + textual ratings
fetch_fact_checks(claim_text, language_code='en', max_results=5)

# Rating map (0.0-1.0)
{'true': 1.0, 'mostly-true': 0.85, 'mixed': 0.5, 'false': 0.0, 'pants-fire': 0.0}

# Retry logic: 3 attempts with exponential backoff
# API limits: 1000 queries/day (free tier)
```

### Translation (src/translate.py)
```python
# Auto-translates non-English to English for model inference
translate_to_english(text, source_lang='es')  # Uses deep-translator

# Disable translation
FAKESCOPE_DISABLE_TRANSLATION=1 streamlit run src/app.py
```

**Important**: DistilBERT classifier is English-only. Non-English text is auto-translated unless disabled.

## Storage Pattern (SQLite)

```python
# src/storage.py - Persistent predictions database
DB_PATH = os.getenv("FAKESCOPE_DB_PATH", "./data/predictions.db")

# Schema: predictions table with migration support
# Columns: id, ts, input_type, url, title, text, model_fake, model_true, google_score, explanation

# Usage
from src.storage import init_db, insert_prediction, fetch_recent
init_db()  # Creates schema if missing
insert_prediction(input_type='text', text='...', model_fake=0.1, model_true=0.9, ...)
recent = fetch_recent(limit=50)  # For dashboard
```

**Deployment**: Mount persistent volume to `/data` in Docker/Fly.io to preserve predictions across restarts.

## Configuration Management

### Environment Variables
```bash
# Model loading
FAKESCOPE_MODEL_DIR=./models/distilbert_fakenews_2stage  # Local or HF repo ID
FAKESCOPE_MODEL_MAX_LENGTH=512

# LLM configuration
FAKESCOPE_LLM_PROVIDER=openai  # openai|gemini|perplexity
FAKESCOPE_OPENAI_MODEL=gpt-4o-mini
FAKESCOPE_GEMINI_MODEL=gemini-2.5-flash
FAKESCOPE_PERPLEXITY_MODEL=sonar-pro

# API keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
GOOGLE_FACTCHECK_API_KEY=...

# Storage
FAKESCOPE_DB_PATH=./data/predictions.db

# Features
FAKESCOPE_DISABLE_TRANSLATION=1  # Disable auto-translation
```

### Config Dataclasses (src/config.py)
```python
@dataclass
class FakeScopeConfig:
    data: DataConfig  # raw_data_paths, encoding, test_size
    preprocessing: PreprocessingConfig  # min_token_length, custom_stopwords
    tfidf: TFIDFConfig  # max_features=5000, ngram_range=(1,2)
    models: ModelConfig  # lr_params, rf_params, xgb_params
    mlflow: MLFlowConfig  # experiment_name, tracking_uri
```

**Usage**: Import singleton `from src.config import config` for centralized settings.

## Testing Strategy

```python
# tests/conftest.py - Shared fixtures
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({'text': ['fake news'], 'class': ['0']})

# tests/test_data_pipeline.py - Data preprocessing
def test_label_normalization()
def test_duplicate_detection()

# tests/test_models.py - Model instantiation
def test_model_config()

# pytest.ini - Coverage threshold 10%
--cov-fail-under=10
```

## Common Pitfalls & Solutions

### Model Not Found Error
```python
# Check 3 locations in order:
# 1. Local directory
ls -la distilbert_fakenews_2stage/config.json

# 2. HF Hub (if MODEL_DIR contains '/')
export FAKESCOPE_MODEL_DIR=enriest/fakescope-distilbert-2stage
export HF_TOKEN=hf_...  # If private repo

# 3. URL normalization
# ✅ https://huggingface.co/enriest/fakescope-distilbert-2stage → enriest/fakescope-distilbert-2stage
```

### Gemini API Errors
```python
# Issue: SDK v1beta instability
# Solution: Use REST API in _gemini_generate_via_rest()

# Model fallback order (hardcoded in src/openai_explain.py)
models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.5-flash-lite']
```

### Class Label Confusion
- Dataset uses **string labels** ('0'/'1') not integers
- '0' = Fake, '1' = True
- Always convert: `df['class'].astype(int)` before model training
- Remove ambiguous labels: `['half-true', 'pants-fire', 'full-flop']` excluded in preprocessing

### Duplicate Leakage Detection
```python
# Check for data leakage before training
hashes = df_news['clean_text'].apply(lambda s: hashlib.md5(s.encode()).hexdigest())
print(f"Unique: {hashes.nunique()} of {len(hashes)}")  # Should be close to 100%
```

### Model Not Loading
```python
# Check for trained models before execution
import os
if not os.path.exists('./distilbert_fakenews_2stage/config.json'):
    print("⚠️ Run Development.ipynb training cells first")
```

## Code Organization Patterns

### Notebook Structure (Development.ipynb)
1. **Data Loading & Merging** (cells 3-6): Combines `alt/News.csv` + `alt 2/New Task.csv`
2. **Preprocessing** (cells 11-13): Stopwords, cleaning, TF-IDF
3. **Baseline Models** (cells 15-23): LogReg, DecisionTree, RandomForest with grid search
4. **Transformer Training** (cells 30-35): DistilBERT standard fine-tuning
5. **2-Stage Training** (cells 39-41): MLM + classification (skip if models exist)
6. **Evaluation & Visualization** (cells 43-50): ROC curves, confusion matrices, attention viz

### File Naming Convention
- `*_fold{1-5}/` → Cross-validation results (5-fold)
- `mlm_results/checkpoint-*/` → Stage 1 MLM checkpoints (every epoch)
- `results_2stage/` → Stage 2 final model checkpoints

## Environment Requirements
```txt
# Key dependencies (from requirements.txt)
torch>=2.3.0               # MPS support for Apple Silicon
transformers>=4.44.2       # DistilBERT + Trainer
datasets>=2.19.0           # HuggingFace Dataset API
openai>=1.52.2             # LLM pipeline integration
google-api-python-client   # Fact Check API
joblib>=1.4.2              # Baseline model persistence
```

## Testing & Validation

### Quick Model Sanity Check
```python
# Run after training to verify model works
from sklearn.metrics import classification_report
preds = trainer.predict(test_dataset)
print(classification_report(preds.label_ids, preds.predictions.argmax(1), digits=4))

# Expected: Accuracy >0.95, F1 >0.94 (both classes)
```

### Pipeline Integration Test
```python
# Test full LLM pipeline (requires API keys)
pipeline = FakeScopePipeline(use_transformer=True)
result = pipeline.run("Sample claim text")
assert result['schema_valid'] == True  # JSON schema validation
```

## Deployment Notes
- **Cache management**: Delete `factcheck_cache.json` to force API refresh
- **Model versioning**: Store `distilbert_fakenews_2stage/` with git-lfs or external storage (>400MB)
- **API rate limits**: Google Fact Check has daily quota; implement retry logic (already present)

## Quick Reference

### Key Files
- **Notebooks**: `notebooks/Development.ipynb` (training), `notebooks/Project.ipynb` (combined)
- **Production**: `src/app.py` (UI), `src/inference.py` (model), `src/openai_explain.py` (LLM)
- **Deployment**: `Dockerfile`, `fly.toml`, `docs/DEPLOYMENT.md`
- **Config**: `src/config.py`, `requirements.txt`, `pytest.ini`, `Makefile`

### Makefile Commands
```bash
make install      # Install deps + NLTK/spacy models
make test         # Run pytest
make coverage     # HTML coverage report
make format       # black + isort
make docker-build # Build container
```

### Model Performance
- **Baseline**: LogReg 92-95%, RF 93-96%, XGBoost 94-97%
- **Standard Transformer**: 97-99% accuracy
- **2-Stage Transformer**: 98-99.5% accuracy (+1-3% boost)

### API Rate Limits
- **Google Fact Check**: 1000/day (free tier)
- **Gemini**: 1500/day free (then $0.005-0.01 per request)
- **OpenAI**: ~$0.01-0.03 per explanation (gpt-4o-mini)
- **Perplexity**: ~$0.01-0.05 per explanation (sonar-pro)

### Documentation
- Project overview: `docs/fakescope-complete.md`
- Deployment: `docs/DEPLOYMENT.md`, `docs/DEPLOY_FLY.md`, `docs/DEPLOY_GCP.md`
- LLM guides: `docs/GEMINI_SETUP.md`, `docs/PERPLEXITY_QUICKSTART.md`, `docs/PROMPT_CUSTOMIZATION.md`
- Features: `docs/ENHANCED_FEATURES.md`, `docs/I18N_GUIDE.md`, `docs/API_STATUS.md`
