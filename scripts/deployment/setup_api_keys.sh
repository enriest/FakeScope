#!/bin/bash
# FakeScope API Keys Setup Script
# Run with: source setup_api_keys.sh

echo "🔑 FakeScope API Keys Setup"
echo "=============================="
echo ""

# Check if .env file exists
if [ -f .env ]; then
    echo "📝 Loading from .env file..."
    source .env
else
    echo "⚠️  No .env file found. Setting keys manually..."
fi

# Set API keys (replace with your actual keys)
echo ""
echo "Setting up API keys..."

# OpenAI (if not set)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set"
    echo "   Get your key from: https://platform.openai.com/api-keys"
else
    echo "✅ OPENAI_API_KEY is set: ${OPENAI_API_KEY:0:10}..."
    export OPENAI_API_KEY
fi

# Gemini (if not set)
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set"
    echo "   Get your key from: https://ai.google.dev/"
else
    echo "✅ GEMINI_API_KEY is set: ${GEMINI_API_KEY:0:10}..."
    export GEMINI_API_KEY
fi

# Perplexity (if not set)
if [ -z "$PERPLEXITY_API_KEY" ]; then
    echo "❌ PERPLEXITY_API_KEY not set"
    echo "   Get your key from: https://www.perplexity.ai/settings/api"
else
    echo "✅ PERPLEXITY_API_KEY is set: ${PERPLEXITY_API_KEY:0:10}..."
    export PERPLEXITY_API_KEY
fi

# Google Fact Check (optional)
if [ -z "$GOOGLE_FACTCHECK_API_KEY" ]; then
    echo "⚠️  GOOGLE_FACTCHECK_API_KEY not set (optional)"
else
    echo "✅ GOOGLE_FACTCHECK_API_KEY is set: ${GOOGLE_FACTCHECK_API_KEY:0:10}..."
    export GOOGLE_FACTCHECK_API_KEY
fi

# Set default provider
if [ -z "$FAKESCOPE_LLM_PROVIDER" ]; then
    echo ""
    echo "⚙️  Setting default provider to: gemini (free tier)"
    export FAKESCOPE_LLM_PROVIDER="gemini"
else
    echo ""
    echo "⚙️  Current provider: $FAKESCOPE_LLM_PROVIDER"
fi

echo ""
echo "=============================="
echo "✨ Setup complete!"
echo ""
echo "To change provider, run:"
echo "  export FAKESCOPE_LLM_PROVIDER=openai     # Use OpenAI"
echo "  export FAKESCOPE_LLM_PROVIDER=gemini     # Use Gemini (FREE)"
echo "  export FAKESCOPE_LLM_PROVIDER=perplexity # Use Perplexity"
echo ""
echo "Then run the app with:"
echo "  python -m streamlit run src/app.py"
echo ""
