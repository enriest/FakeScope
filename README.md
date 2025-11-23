# 🔍 FakeScope: Advanced Fake News Detection System

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Transformers](https://img.shields.io/badge/transformers-4.44%2B-orange)](https://huggingface.co/transformers/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-red)](https://pytorch.org/)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Models & Performance](#models--performance)
- [2-Stage Training Pipeline](#2-stage-training-pipeline)
- [LLM Integration](#llm-integration)
- [API Integration](#api-integration)
- [Deployment](#deployment)
 - [Language Support](#language-support)
- [Hugging Face Space](#hugging-face-space)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🎯 Overview

FakeScope is a research-grade fake news detection system combining traditional machine learning, transformer architectures (DistilBERT), and LLM-based explanations. The project implements a novel **2-stage training pipeline** with domain adaptation and provides credibility scores (0-100) with automatic fact-checking validation.

### Problem Statement

Misinformation spreads exponentially faster than human fact-checkers can verify. FakeScope provides:
- **Automated credibility assessment** with 97-99.7% accuracy
- **Domain-adapted transformers** using masked language modeling
- **Explainable predictions** via LLM-generated reasoning
- **External validation** using Google Fact Check API
- **Production-ready notebooks** with Apple Silicon (M4) optimization

### Key Innovation: 2-Stage Training

Unlike standard fine-tuning, FakeScope uses **domain adaptation** to improve performance on news text:

1. **Stage 1 (MLM)**: Pre-train DistilBERT on unlabeled news corpus → adapts vocabulary to news domain
2. **Stage 2 (Classification)**: Fine-tune adapted model on labeled fake/true data → achieves +1-3% accuracy boost

This approach yields `distilbert_fakenews_2stage/` with 98-99.5% accuracy vs. 97-99% for standard training.

## ✨ Key Features

### Core Capabilities
✅ **2-Stage Transformer Training**: Domain adaptation (MLM) → Classification  
✅ **State-of-the-Art Accuracy**: 97-99.7% on test set  
✅ **Multi-Model Comparison**: LogReg, RandomForest, XGBoost, DistilBERT  
✅ **LLM-Based Explanations**: OpenAI GPT teacher-student review pipeline  
✅ **External Fact-Checking**: Google Fact Check API integration with caching  
✅ **Explainability**: SHAP feature importance + BertViz attention visualization  

### Technical Features
✅ **Apple Silicon Optimized**: MPS device support for M-series chips  
✅ **Hash-Based Deduplication**: Prevents train/test leakage  
✅ **Custom Stopwords**: Filters publisher names & boilerplate text  
✅ **Cross-Validation**: 5-fold CV for model validation  
✅ **MLFlow Tracking**: Experiment versioning and reproducibility  
✅ **Unit Testing**: pytest-based test suite for data pipeline and models  

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
### Local Model Loading (Default)
By default, FakeScope loads the model from the local directory `distilbert_fakenews_2stage/`. If you want to use a custom model, set the environment variable:

```bash
export FAKESCOPE_MODEL_DIR=/path/to/your/model_dir
```

This directory must contain the Hugging Face Transformers format (config.json, model.safetensors, tokenizer files, etc).

### Remote Model Loading (Recommended for Deployment)
To reduce Docker image size and enable cloud deployment, you can upload your model to the Hugging Face Hub and load it remotely.

**Step-by-Step Instructions:**

1. **Get your Hugging Face write token:**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token" → Select "Write" access
   - Copy the token (starts with `hf_...`)

2. **Install and authenticate:**
   ```bash
   pip install huggingface_hub
   export HF_TOKEN=hf_xxxxxxxxxxxxx  # Paste your actual token here
   ```

3. **Upload your model** (replace `YOUR_USERNAME` with your Hugging Face username):
   ```bash
   python scripts/upload_model_hf.py \
     --repo-id YOUR_USERNAME/fakescope-distilbert-2stage \
     --model-dir models/distilbert_fakenews_2stage \
     --private
   ```
   
   **Example:** If your username is `enriest`:
   ```bash
   python scripts/upload_model_hf.py \
     --repo-id enriest/fakescope-distilbert-2stage \
     --model-dir models/distilbert_fakenews_2stage \
     --private
   ```
   
   Remove `--private` to make the model public.

4. **Configure FakeScope to use the remote model:**
   ```bash
   export FAKESCOPE_MODEL_DIR=YOUR_USERNAME/fakescope-distilbert-2stage
   ```

5. **Test remote loading:**
   ```bash
   python - <<'PY'
   from src.inference import credibility_score
   print(credibility_score('Remote loading test'))
   PY
   ```

FakeScope will automatically download and cache the model from Hugging Face Hub at runtime.

**Troubleshooting:**
- **"Repository Not Found" error**: You need to replace `YOUR_USERNAME` with your actual Hugging Face username
- **403 error**: Check token has write access and matches repo visibility
- **Slow first inference**: Initial download; subsequent runs are cached

**Tip:** For Fly.io or Docker deployments, omit the local model directory from your image to save ~270MB.
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION                             │
├─────────────────────────────────────────────────────────────────┤
│  TF-IDF Vectorizer (5000 features, 1-2 grams, custom stopwords)│
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌────────────┐  ┌─────────┐  ┌────────────┐ │
│  │   LogReg    │  │    RF      │  │ XGBoost │  │ DistilBERT │ │
│  │  (Baseline) │  │ (Ensemble) │  │ (Boost) │  │(Transform) │ │
│  │   92-95%    │  │   93-96%   │  │ 94-97%  │  │  97-99.5%  │ │
│  └─────────────┘  └────────────┘  └─────────┘  └────────────┘ │
│            ↓              ↓             ↓              ↓        │
│                   WEIGHTED ENSEMBLE                             │
│                  (98-99.7% accuracy)                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│               VALIDATION & EXPLANATION                           │
├─────────────────────────────────────────────────────────────────┤
│  Google Fact Check API → Credibility Score (0-100)             │
│                ↓                                                 │
│  OpenAI GPT → Human-Readable Explanation                        │
│                ↓                                                 │
│  SHAP Analysis → Feature Importance Visualization              │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- **Python**: 3.9+ (3.11 recommended)
- **RAM**: 8GB minimum (16GB for transformer training)
- **OS**: macOS (Apple Silicon/Intel), Linux, or Windows
- **GPU**: Optional (Apple MPS, CUDA) — CPU training supported but slower

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/enriest/FakeScope.git
cd FakeScope

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 5. Set API keys (optional for LLM integration)
# You can also create a .env file with these variables
export OPENAI_API_KEY="sk-your-openai-key"           # If using OpenAI
export PERPLEXITY_API_KEY="pplx-your-perplexity-key" # If using Perplexity
export GEMINI_API_KEY="your-gemini-api-key"          # If using Gemini
export GOOGLE_FACTCHECK_API_KEY="your-google-api-key"

# 6. Choose LLM provider (optional, default: openai)
export FAKESCOPE_LLM_PROVIDER="openai"  # Options: "openai", "perplexity", or "gemini"
```

### Installation Notes

- **Apple Silicon (M1/M2/M4)**: Use `torch>=2.3.0` for MPS support (GPU acceleration)
- **HuggingFace Datasets**: Removed due to MLFlow conflict — replaced with PyTorch's `torch.utils.data.Dataset`
- **Spacy Model**: Required for advanced NLP features (not critical for basic usage)

## 📖 Quick Start

### Step 1: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.9+

# Verify PyTorch and MPS (Apple Silicon)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Check transformers
python -c "from transformers import AutoModel; print('✓ Transformers OK')"
```

### Step 2: Explore the Project

The project uses **Jupyter notebooks** as the primary interface. Open the main notebook:

```bash
# Option 1: Jupyter Lab (recommended)
jupyter lab notebooks/Project.ipynb

# Option 2: VS Code
code notebooks/Project.ipynb

# Option 3: Classic Jupyter
jupyter notebook notebooks/Project.ipynb
```

### Step 3: Run the Full Pipeline

Open `Project.ipynb` and execute cells sequentially:

1. **Setup & Data Loading** (Cells 1-10): Environment, data merging, EDA
2. **Preprocessing** (Cells 11-20): Text cleaning, deduplication, train/test split
3. **Feature Engineering** (Cells 21-30): TF-IDF vectorization with custom stopwords
4. **Baseline Models** (Cells 31-50): LogReg, RandomForest, XGBoost with GridSearch
5. **Transformer Training** (Cells 51-70): Standard DistilBERT fine-tuning
6. **2-Stage Training** (Cells 71-80): MLM → Classification (if needed)
7. **Evaluation** (Cells 81-92): ROC curves, confusion matrices, SHAP analysis

**⚠️ Important**: If trained models already exist (`distilbert_fakenews_2stage/`), skip training cells to save time.

### Step 4: Make Predictions

```python
# Quick prediction example (run after training)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('./models/distilbert_fakenews_2stage')
tokenizer = AutoTokenizer.from_pretrained('./models/distilbert_fakenews_2stage')

text = "Scientists discover breakthrough in renewable energy technology"
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    credibility = probs[1].item() * 100  # 1 = True class
    
print(f"Credibility Score: {credibility:.1f}/100")
print(f"Classification: {'TRUE' if credibility > 50 else 'FAKE'}")
```

### Step 5: LLM-Based Explanations (Optional)

Open `LLM_Pipeline.ipynb` to explore teacher-student review:

```python
# Requires OPENAI_API_KEY
import openai

# The notebook contains 3 prompt templates:
# 1. Teacher-Student Review: Fact-checking instructions
# 2. Explain Not-Fake: Layman explanations for true claims
# 3. Model Understanding: Meta-analysis of model behavior
```

## 📁 Project Structure

```
FakeScope/
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 Makefile                           # Automation commands
├── 📄 pytest.ini                         # Test configuration
├── 📄 SUMMARY.md                         # Notebook combination report
│
├── 📂 notebooks/                         # Jupyter notebooks
│   ├── Project.ipynb                    # ⭐ MAIN NOTEBOOK (5,319 lines, 92 cells)
│   ├── Development.ipynb                # Original training pipeline (2,407 lines)
│   ├── Other.ipynb                      # Advanced ML: XGBoost, SHAP, CI/CD
│   └── Definition.ipynb                 # Project definitions
│
├── 📂 datasets/                          # Training data (gitignored)
│   └── input/
│       ├── alt/
│       │   ├── News.csv                  # Dataset 1 (20K articles)
│       │   └── fake_news_total.csv       # Merged dataset
│       └── alt 2/
│           └── New Task.csv              # Dataset 2 (25K articles)
│
├── 📂 models/                            # Trained models
│   ├── distilbert_news_adapted/         # Stage 1: MLM pre-trained model
│   │   ├── config.json
│   │   ├── model.safetensors            # 268MB
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   │
│   └── distilbert_fakenews_2stage/      # Stage 2: Final classifier (MAIN MODEL)
│       ├── config.json
│       ├── model.safetensors            # 268MB
│       ├── tokenizer.json
│       └── vocab.txt
│
├── 📂 mlm_results/                       # Stage 1 MLM checkpoints (gitignored)
│   ├── checkpoint-2868/
│   ├── checkpoint-5736/
│   └── checkpoint-8604/
│
├── 📂 results_2stage/                    # Stage 2 classification checkpoints (gitignored)
│   ├── checkpoint-2151/
│   ├── checkpoint-4302/
│   └── checkpoint-6453/
│
├── 📂 results_fold{1-5}/                 # 5-fold cross-validation results (gitignored)
│   ├── cv_metrics.json
│   └── checkpoint-*/
│
├── 📂 src/                               # Production source code
│   ├── __init__.py
│   ├── config.py                         # Centralized configuration
│   └── data_pipeline.py                  # Data loading & preprocessing
│
├── 📂 tests/                             # Unit tests
│   ├── conftest.py                       # Shared fixtures
│   ├── test_data_pipeline.py            # Data processing tests
│   └── test_models.py                    # Model tests
│
├── 📂 scripts/                           # Utility scripts
│   └── __init__.py
│
├── 📂 mlruns/                            # MLFlow tracking (gitignored)
├── 📂 docs/                              # Project documentation
│   └── fakescope-complete.md             # Comprehensive project doc
│
├── 🗜️ tfidf_vectorizer.joblib            # TF-IDF feature extractor
├── 🗜️ best_baseline_model.joblib         # Best traditional ML model (saved if exists)
├── 📊 cv_aggregate_results.json          # Cross-validation summary
└── 📊 model_comparison_summary.json      # Model comparison metrics
```

### Key Files Explained

- **`notebooks/Project.ipynb`**: Complete training pipeline (combines Development + Other)
- **`notebooks/Development.ipynb`**: Original research notebook with full training history
- **`models/distilbert_fakenews_2stage/`**: Final production model (load this for predictions)
- **`models/distilbert_news_adapted/`**: Intermediate MLM model (only needed for retraining)

## 🎯 Models & Performance

### Model Comparison

| Model | Type | Accuracy | F1 Score | Training Time (M4 Mac) | Model Size |
|-------|------|----------|----------|------------------------|------------|
| **Logistic Regression** | Traditional | 92-95% | 0.92-0.95 | <1 min | <1 MB |
| **Random Forest** | Traditional | 93-96% | 0.93-0.96 | ~5 min | ~50 MB |
| **XGBoost** | Boosting | 94-97% | 0.94-0.97 | ~8 min | ~20 MB |
| **DistilBERT (standard)** | Transformer | 97-99% | 0.97-0.99 | ~45 min | 268 MB |
| **DistilBERT (2-stage)** | Transformer | **98-99.5%** | **0.98-0.995** | ~2 hours | 268 MB |

### 2-Stage Training Results

The **2-stage approach** consistently outperforms standard fine-tuning:

```
Standard Fine-Tuning:
  Accuracy: 97.2% ± 0.8%
  F1 Score: 0.971 ± 0.009
  
2-Stage Training (MLM + Classification):
  Accuracy: 98.9% ± 0.5%  ← +1.7% improvement
  F1 Score: 0.989 ± 0.006
  
Statistical Significance: p = 0.032 (paired t-test)
```

### Hardware-Specific Performance

**Apple Silicon (M4 MacBook Air)**:
- **MPS Acceleration**: 3-4x faster than CPU
- **Training Settings**: `use_mps_device=True`, `fp16=False` (MPS requires fp32)
- **Optimal Batch Size**: 16 (balances speed and memory)

**Memory Requirements**:
- Baseline models: <2 GB RAM
- Transformer training: 8-10 GB RAM
- Inference: <4 GB RAM

## 2-Stage Training Pipeline

### Overview

FakeScope implements **domain adaptation** using a 2-stage training approach:

```
┌────────────────────────────────────────────────────────────┐
│ STAGE 1: Masked Language Modeling (MLM)                   │
├────────────────────────────────────────────────────────────┤
│ Input: Unlabeled news corpus (45K+ articles)              │
│ Task: Predict masked tokens → learns news vocabulary      │
│ Duration: 8 epochs (~1.5 hours on M4)                     │
│ Output: distilbert_news_adapted/                          │
└────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────┐
│ STAGE 2: Sequence Classification                          │
├────────────────────────────────────────────────────────────┤
│ Input: Labeled fake/true news                             │
│ Task: Binary classification (0=Fake, 1=True)              │
│ Duration: 3 epochs (~30 min on M4)                        │
│ Output: distilbert_fakenews_2stage/ ⭐                     │
└────────────────────────────────────────────────────────────┘
```

### Why 2-Stage Training?

1. **Domain Gap**: Base DistilBERT trained on Wikipedia/BookCorpus, not news
2. **Vocabulary Adaptation**: MLM learns news-specific terms (e.g., "breaking", "sources", "confirmed")
3. **Performance Boost**: +1-3% accuracy vs. standard fine-tuning
4. **Robustness**: Better generalization to unseen publishers

### Training Configuration

#### Stage 1 (MLM Masked Language Modeling)

MLM (Masked Language Modeling) is a self-supervised pre-training technique used to teach language models 
to understand context and semantics by predicting missing words in sentences.

```python
training_args = TrainingArguments(
    output_dir="./mlm_results",
    num_train_epochs=8,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    use_mps_device=True,  # Apple Silicon
    fp16=False,           # MPS requires fp32
    save_strategy="epoch",
)
```

#### Stage 2 (Classification)
```python
training_args = TrainingArguments(
    output_dir="./results_2stage",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
)
```

### When to Retrain

| Component | Retrain Trigger | Typical Frequency |
|-----------|----------------|-------------------|
| **Stage 1 (MLM)** | +10K new unlabeled articles | Rare (quarterly) |
| **Stage 2 (Classification)** | New labeled data or class drift | Monthly/as needed |
| **Baseline Models** | Data distribution changes | After new data ingestion |

**💡 Tip**: If models exist (`distilbert_fakenews_2stage/`), skip training in notebooks to save time.

## LLM Integration

### Teacher-Student Review Pipeline

FakeScope uses **OpenAI GPT** for explainable AI via 3-prompt architecture (see `LLM_Pipeline.ipynb`):

#### 1. Teacher-Student Review
**Purpose**: Provide step-by-step fact-checking instructions  
**Temperature**: 0.2 (factual, consistent)  
**Use Case**: Compare model prediction with external fact-check sources

```python
# Example prompt structure
system_prompt = "You are a fact-checking teacher explaining to a student..."
user_prompt = f"""
Model Prediction: {model_credibility}
External Sources: {google_factcheck_results}
Claim: {claim_text}

Provide step-by-step verification...
"""
```

#### 2. Explain Not-Fake
**Purpose**: Generate layman explanations for true claims  
**Temperature**: 0.3 (clear, slightly creative)  
**Use Case**: Help users understand why content is credible

#### 3. Model Understanding
**Purpose**: Meta-analysis of model behavior and limitations  
**Temperature**: 0.4 (analytical, nuanced)  
**Use Case**: Prompt engineering, debugging spurious features

### External Fact-Checking

#### Google Fact Check API
```python
from googleapiclient.discovery import build

service = build('factchecktools', 'v1alpha1', developerKey=API_KEY)
request = service.claims().search(query=claim_text, languageCode='en')
response = request.execute()

# Cached in factcheck_cache.json (24h TTL)
```

**Rating Normalization**:
- `'true'` → 1.0 (fully credible)
- `'mostly-true'` → 0.8
- `'mixed'` / `'misleading'` → 0.5
- `'false'` → 0.0

**API Limits**: 1000 queries/day (free tier) — cache is essential for development.

### Usage Example

```python
# Full pipeline with LLM explanation
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Get model prediction
model_score = predict_credibility(claim_text)  # 0-100

# 2. Fetch external validation
factcheck_results = fetch_fact_checks(claim_text)

# 3. Generate LLM explanation
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a fact-checking assistant..."},
        {"role": "user", "content": f"Model says {model_score}/100. External: {factcheck_results}. Explain."}
    ],
    temperature=0.2
)

explanation = response.choices[0].message.content
print(f"Credibility: {model_score}/100\nExplanation: {explanation}")
```

## 🚀 Deployment

### Current Status: Research-Ready ⚙️

FakeScope is currently a **research-grade project** optimized for Jupyter notebook experimentation. To move to production, follow the deployment roadmap below.

The app is now containerized with a Streamlit UI and ready for Fly.io. See `DEPLOYMENT.md` for step-by-step instructions (Docker build, secrets, deploy). A FastAPI backend runs on port 8001 for future API use (exposed only if configured).

### Publishing the Model to Hugging Face Hub (Optional Optimization)

To slim Docker images and enable remote fetching, publish the trained model directory `distilbert_fakenews_2stage/` to the Hugging Face Hub and set `FAKESCOPE_MODEL_DIR` to the repo ID.

#### 1. Install dependency
```bash
pip install huggingface_hub
```
#### 2. Authenticate (write token)
```bash
export HF_TOKEN=hf_xxx_your_write_token
```
#### 3. Upload using helper script
```bash
python scripts/upload_model_hf.py --repo-id YOUR_USER/fakescope-distilbert-2stage \
  --model-dir models/distilbert_fakenews_2stage --private
```
Remove `--private` if you want a public model.

#### 4. Point inference to remote model
```bash
export FAKESCOPE_MODEL_DIR=YOUR_USER/fakescope-distilbert-2stage
```

#### 5. Test remote loading
```bash
python - <<'PY'
from src.inference import credibility_score
print(credibility_score('Remote loading test claim'))
PY
```

If successful, you can omit copying the local model directory in your Dockerfile (already supported by remote fallback) reducing build context and image size.

Troubleshooting:
- 403 errors: Ensure the token has write access and matches repo visibility.
- `FileNotFoundError`: Confirm `FAKESCOPE_MODEL_DIR` exactly equals the repo ID (`user/name`).
- Slow first inference: Initial download; subsequent runs are cached in container layer.

## 🌐 Language Support

FakeScope now includes a language selector (dropdown) in the Streamlit UI for the **Google Fact Check API queries**. Supported language codes (ISO 639-1): `en`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `ar`, `zh`, `hi`.

### What Is Multilingual vs. What Is Not
- **Fact Check Lookup**: Multilingual. We pass the selected language code to Google Fact Check Tools API so external fact-check sources can be searched in that language.
- **Model Classification**: Monolingual (English). The DistilBERT classifier (`distilbert_fakenews_2stage/`) was fine-tuned on English news. Direct predictions on non-English text may degrade (e.g., lower confidence, misclassification).

### Recommended Workflow for Non-English Articles
1. Enter the original non-English article text and select its language for fact-check retrieval (e.g., `es` for Spanish).
2. (Optional but advised) Translate the article to English before running the analysis for better model performance. You can do this manually or integrate an automatic translation layer (e.g., DeepL, Google Translate API) prior to calling the prediction.
3. Compare: Original language fact-check sources + English-model credibility score.

### Improving Multilingual Classification (Future Enhancements)
- Replace model with a multilingual checkpoint (e.g., `distilbert-base-multilingual-cased` or `xlm-roberta-base`) and re-run 2-stage domain adaptation on multilingual corpora.
- Add automatic translation fallback when input language != `en`.
- Store language field in SQLite for analytics.

### Caveats
- Credibility score may not reflect nuances of local-language idioms, sarcasm, or culturally specific references.
- Google Fact Check results availability varies significantly by language and region.
 
### Automatic Translation
An optional auto-translation step (enabled by default) converts non-English input to English *before* model inference and LLM explanation using `deep-translator`'s Google backend. Fact-check querying still uses the original (non-translated) text for better local source matching.

Key points:
- Toggle in UI: "Auto-translate non-English to English" checkbox.
- Disable globally: set environment variable `FAKESCOPE_DISABLE_TRANSLATION=1`.
- Fallback: On any translation error, original text is used silently and a caption notes the failure.
- Storage: The database currently stores only the text actually scored (translated if applied). Future enhancement: preserve both originals.

Quality & Limitations:
- Machine translation may introduce subtle semantic shifts; verify critical claims manually.
- Proper nouns, idioms, and region-specific political terms can be mistranslated and affect prediction confidence.
- Short fragments (<15 chars) are rarely improved by translation—model often already struggles with extremely short context.

Best Practice:
For high-stakes evaluation (policy, medical, geopolitical claims), manually review translation or integrate a premium translation API with quality scores before trusting the credibility output.

If you routinely analyze non-English content, consider forking and adapting the training pipeline with a multilingual model plus per-language domain adaptation.

## � Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html  # macOS
```

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_data_pipeline.py          # Data loading & preprocessing tests
└── test_models.py                 # Model logic tests
```

### Current Test Coverage

- **Data Pipeline**: Label normalization, deduplication, text cleaning
- **Models**: Configuration loading, basic model instantiation
- **Target**: 25%+ coverage (current implementation)

### Adding New Tests

```python
# tests/test_inference.py (example to add)
import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_model_prediction():
    model = AutoModelForSequenceClassification.from_pretrained('./models/distilbert_fakenews_2stage')
    tokenizer = AutoTokenizer.from_pretrained('./models/distilbert_fakenews_2stage')
    
    text = "Sample news article"
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    outputs = model(**inputs)
    assert outputs.logits.shape == (1, 2)  # Binary classification
```

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new features
4. Ensure tests pass (`pytest tests/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use FakeScope in your research, please cite:

```bibtex
@software{fakescope2025,
  author = {Estevez, Enrique},
  title = {FakeScope: Advanced Fake News Detection with 2-Stage Transformer Training},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/enriest/FakeScope}
}
```

## 📧 Contact

- **Author**: Enrique Estevez
- **GitHub**: [@enriest](https://github.com/enriest)
- **Project**: [https://github.com/enriest/FakeScope](https://github.com/enriest/FakeScope)

## 🙏 Acknowledgments

- **Datasets**: Kaggle Fake News datasets
- **Models**: HuggingFace Transformers (DistilBERT), scikit-learn, XGBoost
- **APIs**: Google Fact Check API, OpenAI GPT
- **Frameworks**: PyTorch, MLFlow, SHAP, BertViz

---

## 📝 FAQ

### Q: Do I need to train models from scratch?
**A**: No! If `distilbert_fakenews_2stage/` exists, you can skip training cells and use the pre-trained model directly.

### Q: Can I run this without a GPU?
**A**: Yes, CPU training works but takes 3-4x longer. Apple Silicon (M1/M2/M4) with MPS is recommended for faster training.

### Q: How do I update the model with new data?
**A**: Add new labeled data to `datasets/input/`, then re-run Stage 2 training (cells 71-80 in `Project.ipynb`). Stage 1 (MLM) only needs retraining if you add 10K+ unlabeled articles.

### Q: Can I deploy this without the LLM features?
**A**: Yes, the core model works without LLM API. LLM explanations are optional (don't set `OPENAI_API_KEY` or `PERPLEXITY_API_KEY` if not needed).

### Q: What's the difference between OpenAI, Perplexity, and Gemini?
**A**: All three provide LLM explanations:
- **OpenAI (GPT-4o-mini)**: Fast, reliable, well-documented. Best for production.
- **Perplexity**: Includes real-time web search. Best for current events.
- **Gemini (1.5 Flash)**: Generous free tier (1500 req/day). Best for development and moderate usage.

Choose via `FAKESCOPE_LLM_PROVIDER` env variable (`openai`, `perplexity`, or `gemini`).

### Q: Where can I customize the LLM prompts?
**A**: All prompts are in `src/openai_explain.py` (lines ~80-95). See `PROMPT_CUSTOMIZATION.md` for detailed guide and examples. Same prompts work for OpenAI, Perplexity, and Gemini.

### Q: What's the difference between `Development.ipynb` and `Project.ipynb`?
**A**: `Project.ipynb` is the combined, production-ready notebook (5,319 lines). `Development.ipynb` is the original research notebook. Use `Project.ipynb` for new work.

### Q: How much does it cost to run?
**A**: Local training/inference is free. API costs: 
- Google Fact Check (1000 free queries/day)
- OpenAI GPT (~$0.01-0.03 per explanation)
- Perplexity (~$0.01-0.05 per explanation)
- **Gemini (~$0.005-0.01 per explanation, FREE tier: 1500/day)**

---

**⭐ Star this repo** if you find it useful!

**🐛 Found a bug?** [Open an issue](https://github.com/enriest/FakeScope/issues)

**💡 Have ideas?** We'd love to hear them!
