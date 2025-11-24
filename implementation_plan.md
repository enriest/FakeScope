# Implementation Plan - Increase Code Coverage

## Goal
Increase project code coverage to approximately 25% by adding unit tests for core utility and logic modules.

## Proposed Changes

### Tests

#### [NEW] [test_utils.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_utils.py)
- Test `extract_text_from_url` with mocked `requests.get`.
- Test success cases (article tag, paragraphs).
- Test failure cases (404, exceptions).

#### [NEW] [test_translate.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_translate.py)
- Test `translate_to_english`.
- Mock `GoogleTranslator`.
- Test bypass conditions (env var, already English).
- Test failure handling.

#### [NEW] [test_inference.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_inference.py)
- Test `_normalize_repo_id`.
- Mock `AutoTokenizer` and `AutoModelForSequenceClassification`.
- Test `predict_proba` and `credibility_score`.
- Test error handling (empty text).

#### [NEW] [test_storage.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_storage.py)
- Use `sqlite3` in-memory DB or temp file.
- Test `ensure_schema`, `insert_prediction`, `fetch_recent`.

#### [NEW] [test_factcheck.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_factcheck.py)
- Test `_normalize_rating`.
- Mock `requests.get` for `fetch_fact_checks`.
- Test `aggregate_google_score`.

#### [NEW] [test_api.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_api.py)
- Use `TestClient` from `fastapi.testclient`.
- Test `/healthz`.
- Test `/predict` with mocked dependencies.

## Verification Plan
- Run `make coverage` (or `pytest --cov=src`) to verify coverage increase.
- Ensure all new tests pass.
