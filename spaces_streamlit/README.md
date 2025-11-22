---
title: FakeScope
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
---

# FakeScope – AI-Powered Fake News Detector

FakeScope combines machine learning (DistilBERT), external fact-checking APIs (Google Fact Check), and LLM explanations (OpenAI/Perplexity/Gemini) to detect fake news and provide credibility scores.

## Features

- **ML Model**: Fine-tuned DistilBERT on 40k+ news articles
- **Fact-Checking**: Google Fact Check API integration
- **LLM Explanations**: OpenAI GPT-4o-mini, Perplexity, or Gemini for human-readable analysis
- **URL Support**: Automatic article extraction from URLs
- **Credibility Score**: 0-100 scale with detailed probability breakdown
- **Dashboard**: Track prediction history with visualizations
- **Persistent Storage**: SQLite database for history tracking

## UI Features

This Space uses the full **Streamlit interface** with:
- 📊 Two-tab layout (Predict + Dashboard)
- 📈 Prediction history with charts
- 🔗 URL text extraction
- 💾 Persistent database
- 🎨 Custom styling

## Model Details

- **Architecture**: DistilBERT (66M parameters)
- **Training**: 2-stage domain adaptation on news corpus
- **Dataset**: Fake/True news articles (balanced)
- **Accuracy**: ~95% on test set
- **Model Hub**: [enri-est/fakescope-distilbert-2stage](https://huggingface.co/enri-est/fakescope-distilbert-2stage)

## Configuration

This Space requires API keys (set in Settings → Repository secrets):

- **Required**:
  - `GOOGLE_FACTCHECK_API_KEY` - For fact-check verification
  - `FAKESCOPE_MODEL_DIR` - Set to `enri-est/fakescope-distilbert-2stage`

- **LLM Provider** (choose one):
  - `GEMINI_API_KEY` + `FAKESCOPE_LLM_PROVIDER=gemini` (recommended, free tier)
  - `OPENAI_API_KEY` + `FAKESCOPE_LLM_PROVIDER=openai`
  - `PERPLEXITY_API_KEY` + `FAKESCOPE_LLM_PROVIDER=perplexity`

## Usage

1. Navigate to the **Predict** tab
2. Enter a news article URL, or paste text directly
3. Click "Run Analysis"
4. View credibility score, fact-checks, and LLM explanation
5. Check the **Dashboard** tab to see prediction history

## Limitations

- Model trained on English news articles
- Performance may vary on social media posts or non-news text
- External fact-checkers may not cover all claims
- LLM explanations are interpretive, not definitive

## Repository

GitHub: [enriest/FakeScope](https://github.com/enriest/FakeScope)

## License

MIT License
