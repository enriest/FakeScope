# Hugging Face Space Variant

This folder contains a minimal Gradio app to run FakeScope on Hugging Face Spaces.

## Steps

1) Create a new Space at https://huggingface.co/spaces (Gradio template).

2) Push these files as the Space repo contents:

- `app.py`
- `requirements.txt`
- Copy your `distilbert_fakenews_2stage/` folder into the Space (recommended via Git LFS or upload via web UI).
- Copy the minimal subset of `src/` used by the app:
  - `src/inference.py`
  - `src/factcheck.py`
  - `src/openai_explain.py`
  - `src/utils.py`

3) Set secrets in the Space Settings → Variables & secrets:

- `OPENAI_API_KEY`
- `GOOGLE_FACTCHECK_API_KEY`

4) Save and restart the Space. The UI provides URL/title/text inputs, model score, Google aggregate, and an explanation.

Notes:
- The Space uses CPU. DistilBERT inference should be responsive, but initial load may take ~10s.
- If you prefer to fetch the model at runtime, modify `src/inference.py` to download from HF Hub and cache in `/home/user/.cache`.
