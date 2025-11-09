# FakeScope AI Coding Agent Instructions

## Project Overview
**FakeScope** is a fake news detection system combining ML/NLP models, transformer architectures (DistilBERT), and external fact-checking APIs to provide credibility scores (0-100) with automatic explanations. The project targets research-grade accuracy with production-ready LLM integration.

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

### Running the Full Pipeline
```bash
# 1. Environment setup (zsh on macOS)
export OPENAI_API_KEY="<key>"
export GOOGLE_FACTCHECK_API_KEY="<key>"
source .venv/bin/activate

# 2. Train models (if not already trained)
# Open Development.ipynb and run cells in order:
# - Data loading → Preprocessing → Baseline models → Transformer training

# 3. Run LLM pipeline
# Open LLM_Pipeline.ipynb and execute example cells
```

### Testing Model Predictions
```python
# Quick test without notebooks
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('./distilbert_fakenews_2stage')
tokenizer = AutoTokenizer.from_pretrained('./distilbert_fakenews_2stage')

inputs = tokenizer("Sample news text", return_tensors='pt', truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    print(f"Fake: {probs[0]:.3f}, True: {probs[1]:.3f}")
```

## External API Integration

### Google Fact Check API
```python
# Lives in Development.ipynb (cell #VSC-6f36943d)
# Cached results in factcheck_cache.json (24h TTL)

def fetch_fact_checks(claim_text, language_code='en', max_results=5):
    # Returns list of dicts with keys: claim_text, textual_rating, title, url, publisher, review_date
    # Uses retry logic (3 attempts) and exponential backoff
    pass
```

**Rating normalization**: Maps textual ratings ('true', 'false', 'mixed', 'misleading') to float scores (0.0-1.0). See `_RATING_MAP` dict in code.

### LLM Review Pipeline (3 Prompts)
Located in `LLM_Pipeline.ipynb`:
1. **Teacher-Student Review**: Step-by-step fact-checking instructions comparing model vs. external sources
2. **Explain Not-Fake**: Layman explanation when evidence suggests claim is true
3. **Model Understanding**: Meta-analysis of model behavior, spurious features, prompt-engineering tips

**Why 3 separate prompts**: Different temperature/token settings optimize each task (review=0.2, explain=0.3, introspect=0.4).

## Common Pitfalls & Solutions

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

## References
- Project doc: `Documents/fakescope-complete.md`
- Main notebook: `Development.ipynb` (2407 lines, full pipeline)
- LLM integration: `LLM_Pipeline.ipynb` (teacher-student prompts)
