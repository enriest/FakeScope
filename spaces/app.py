import os
from typing import Any, Dict, List, Optional

import gradio as gr

from src.inference import predict_proba
from src.factcheck import fetch_fact_checks, aggregate_google_score
from src.openai_explain import generate_explanation
from src.utils import extract_text_from_url


def run_pipeline(url: str, title: str, text: str) -> Dict[str, Any]:
    input_text = text or title or ""
    if url and not input_text:
        extracted = extract_text_from_url(url)
        if extracted:
            input_text = extracted

    if not input_text:
        return {"error": "Provide URL, title, or text"}

    scores = predict_proba(input_text)
    cred = scores["true"] * 100.0

    g_items = fetch_fact_checks(title or input_text)
    g_score = aggregate_google_score(g_items)

    explanation = generate_explanation(
        input_text=input_text,
        model_scores=scores,
        google_items=g_items,
        google_score=g_score,
    )

    items_view = [
        {
            "rating": it.get("textual_rating"),
            "publisher": it.get("publisher"),
            "url": it.get("url"),
        }
        for it in g_items
    ]

    return {
        "credibility": round(cred, 2),
        "prob_true": round(scores["true"], 4),
        "prob_fake": round(scores["fake"], 4),
        "google_score": None if g_score is None else round(g_score, 3),
        "fact_checks": items_view,
        "explanation": explanation or "(OpenAI not configured)",
        "used_url_text": bool(url and not text and input_text),
    }


with gr.Blocks(title="FakeScope – Fake News Detector") as demo:
    gr.Markdown("# FakeScope – Fake News Detector")
    with gr.Row():
        url = gr.Textbox(label="Article URL (optional)")
        title = gr.Textbox(label="Title (optional)")
    text = gr.Textbox(label="Article text or claim", lines=8)

    run_btn = gr.Button("Run Analysis")

    with gr.Row():
        cred = gr.Number(label="Model Credibility (0-100)")
        p_true = gr.Number(label="Prob True")
        p_fake = gr.Number(label="Prob Fake")
        gscore = gr.Number(label="Google Score (0-1)")

    facts = gr.Dataframe(headers=["rating", "publisher", "url"], label="Fact Check Results")
    expl = gr.Markdown(label="Explanation")

    def _submit(u, t, x):
        out = run_pipeline(u, t, x)
        if "error" in out:
            return 0, 0, 0, None, [], out["error"]
        return (
            out["credibility"],
            out["prob_true"],
            out["prob_fake"],
            out["google_score"],
            out["fact_checks"],
            out["explanation"],
        )

    run_btn.click(_submit, inputs=[url, title, text], outputs=[cred, p_true, p_fake, gscore, facts, expl])

if __name__ == "__main__":
    demo.launch()
