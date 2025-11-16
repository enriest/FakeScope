from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import predict_proba, credibility_score
from src.factcheck import fetch_fact_checks, aggregate_google_score


app = FastAPI(title="FakeScope API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str
    include_factcheck: bool = True


class PredictResponse(BaseModel):
    credibility: float
    probs: Dict[str, float]
    google_score: Optional[float] = None


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    probs = predict_proba(req.text)
    cred = credibility_score(req.text)

    gscore: Optional[float] = None
    if req.include_factcheck:
        items = fetch_fact_checks(req.text)
        gscore = aggregate_google_score(items)

    return PredictResponse(credibility=cred, probs=probs, google_score=gscore)
