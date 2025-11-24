# Code Coverage Increase Walkthrough

## Overview
Successfully increased code coverage from **5.83%** to **24.13%** by adding comprehensive unit tests for core modules.

## New Test Files Created

### [test_utils.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_utils.py)
- Tests for `extract_text_from_url` with mocked HTTP requests
- Coverage: **95%** (21/22 lines)

### [test_translate.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_translate.py)
- Tests for `translate_to_english` with mocked GoogleTranslator
- Coverage: **80%** (24/30 lines)

### [test_inference.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_inference.py)
- Tests for model inference functions
- Coverage: **65%** (32/49 lines)

### [test_storage.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_storage.py)
- Tests for database operations with temporary SQLite DB
- Coverage: **95%** (35/37 lines)

### [test_factcheck.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_factcheck.py)
- Tests for Google Fact Check API integration
- Coverage: **88%** (49/56 lines)

### [test_api.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_api.py)
- Tests for FastAPI endpoints using TestClient
- Coverage: **100%** (25/25 lines)

### [test_openai_explain.py](file:///Users/enriqueestevezalvarez/Library/Mobile Documents/com~apple~CloudDocs/Final Project/FakeScope/FakeScope/tests/test_openai_explain.py)
- Tests for LLM explanation generation
- Coverage: **43%** (63/148 lines)

## Test Results

All **64 tests** are passing:
- 27 original tests (data pipeline + models)
- 37 new tests (utils, translate, inference, storage, factcheck, api, openai_explain)

## Coverage Summary

| Module | Coverage | Lines Covered |
|--------|----------|---------------|
| `src/api.py` | 100% | 25/25 |
| `src/config.py` | 100% | 48/48 |
| `src/storage.py` | 95% | 35/37 |
| `src/utils.py` | 95% | 20/21 |
| `src/factcheck.py` | 88% | 49/56 |
| `src/translate.py` | 80% | 24/30 |
| `src/inference.py` | 65% | 32/49 |
| `src/openai_explain.py` | 43% | 63/148 |
| `src/data_pipeline.py` | 33% | 31/94 |
| **TOTAL** | **24.13%** | **327/1355** |

> [!NOTE]
> The target of 25% was nearly achieved at 24.13%. Further coverage can be added by testing the Streamlit app modules (`app.py`, `app_enhanced.py`) which are currently at 0% coverage.
