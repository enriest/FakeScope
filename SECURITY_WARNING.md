# ⚠️ SECURITY WARNING ⚠️

## Exposed API Key Detected!

An OpenAI API key has been found in `Definition.ipynb`. This is a **critical security issue**.

### Immediate Actions Required:

1. **Revoke the exposed key immediately**:
   - Go to https://platform.openai.com/api-keys
   - Find the key starting with `sk-proj-bP2nEZR5...`
   - Click the trash icon to delete it
   - Generate a new key

2. **Remove the key from the notebook**:
   - Open `Definition.ipynb`
   - Delete or redact the cell containing the API key
   - Replace with: `OPEN AI API KEY = os.getenv("OPENAI_API_KEY")`

3. **Check Git history**:
   ```bash
   # If this file was committed to Git, the key is in history!
   git log -- Definition.ipynb
   
   # If committed, you need to purge from history
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch Definition.ipynb' \
     --prune-empty --tag-name-filter cat -- --all
   ```

4. **Monitor for unauthorized usage**:
   - Check OpenAI usage dashboard: https://platform.openai.com/usage
   - Look for unexpected requests
   - Set up usage alerts

### Never Commit API Keys!

**API keys found in**:
- ❌ `Definition.ipynb` (cell with "OPEN AI API KEY =")

**These keys should NEVER be in code**:
- OpenAI API keys (`sk-...`)
- Perplexity API keys (`pplx-...`)
- Google API keys (`AI...`)
- Gemini API keys

### Correct Way to Handle API Keys:

#### 1. Environment Variables (Local Development)
```bash
# In ~/.zshrc or ~/.bashrc
export OPENAI_API_KEY="sk-your-key-here"
export PERPLEXITY_API_KEY="pplx-your-key-here"
export GEMINI_API_KEY="your-gemini-key-here"
export GOOGLE_FACTCHECK_API_KEY="your-google-key-here"
```

#### 2. In Code (Load from Environment)
```python
import os

# ✅ CORRECT
api_key = os.getenv("OPENAI_API_KEY")

# ❌ WRONG
api_key = "sk-proj-..."
```

#### 3. For Notebooks
```python
# Cell 1: Load from environment
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set! Run: export OPENAI_API_KEY='your-key'")
```

#### 4. Production (Fly.io, HF Spaces)
```bash
# Fly.io
flyctl secrets set OPENAI_API_KEY="sk-..."

# Hugging Face Spaces
# Settings → Repository secrets → Add OPENAI_API_KEY
```

### .gitignore Protection

The following files are already protected by `.gitignore`:
```gitignore
# API Keys & Secrets (CRITICAL - Never commit!)
.env
.env.local
*.key
*_key.txt
secrets.json
config/secrets.json
```

But **notebooks** (.ipynb) are NOT in `.gitignore` by default, so you must manually ensure they don't contain keys!

### Checklist Before Every Commit:

- [ ] No API keys in .ipynb files
- [ ] No API keys in .py files
- [ ] No API keys in .md files
- [ ] No API keys in environment examples (use placeholders)
- [ ] Secrets stored in environment variables only

### Safe Placeholder Examples:

```bash
# ✅ Safe to commit
export OPENAI_API_KEY="sk-your-openai-key-here"
export GEMINI_API_KEY="your-gemini-key-here"

# ❌ Never commit
export OPENAI_API_KEY="sk-proj-actual-key-1234567890"
```

### What If My Key Was Already Pushed to GitHub?

1. **Revoke the key immediately** (see step 1 above)
2. **Remove from Git history**:
   ```bash
   # Use BFG Repo-Cleaner (faster than git-filter-branch)
   brew install bfg  # macOS
   
   # Clone a fresh copy
   git clone --mirror https://github.com/enriest/FakeScope.git
   cd FakeScope.git
   
   # Remove the file from history
   bfg --delete-files Definition.ipynb
   
   # Clean up
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   
   # Force push (destructive!)
   git push --force
   ```

3. **Alternative**: Make the repository private if it's public

4. **Rotate all other keys** as a precaution

### Prevention Going Forward:

1. **Use pre-commit hooks**:
   ```bash
   # Install
   pip install pre-commit
   
   # Create .pre-commit-config.yaml
   cat > .pre-commit-config.yaml << 'EOF'
   repos:
   - repo: https://github.com/Yelp/detect-secrets
     rev: v1.4.0
     hooks:
     - id: detect-secrets
       args: ['--baseline', '.secrets.baseline']
   EOF
   
   # Install hooks
   pre-commit install
   ```

2. **Use environment-specific files**:
   - Keep `.env.example` with placeholders (safe to commit)
   - Copy to `.env` with real keys (in .gitignore)

3. **Regular audits**:
   ```bash
   # Search for potential keys in all files
   grep -r "sk-" . --include="*.py" --include="*.ipynb"
   grep -r "pplx-" . --include="*.py" --include="*.ipynb"
   grep -r "AIza" . --include="*.py" --include="*.ipynb"
   ```

---

## Summary

✅ **FakeScope now supports 3 LLM providers:**
- OpenAI (GPT-4o-mini)
- Perplexity (Llama 3.1 Sonar)
- Google Gemini (1.5 Flash) ← **FREE tier available!**

⚠️ **But first**: Secure your API keys!

1. Revoke exposed OpenAI key
2. Clean Definition.ipynb
3. Use environment variables only
4. Check Git history
5. Set up key rotation schedule

**Need help?** See `GEMINI_SETUP.md` for secure configuration examples.
