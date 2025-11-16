import os
from typing import Optional

import streamlit as st

from src.inference import credibility_score, predict_proba
from src.factcheck import fetch_fact_checks, aggregate_google_score
from src.openai_explain import generate_explanation
from src.storage import init_db, insert_prediction, fetch_recent
from src.utils import extract_text_from_url


APP_TITLE = "FakeScope – Fake News Detector"


def _predict_flow(input_text: str, url: Optional[str], title: Optional[str]):
    model_scores = predict_proba(input_text)
    cred = model_scores["true"] * 100.0

    google_items = fetch_fact_checks(title or input_text)
    g_score = aggregate_google_score(google_items)

    explanation = generate_explanation(
        input_text=input_text,
        model_scores=model_scores,
        google_items=google_items,
        google_score=g_score,
    )

    # Persist
    try:
        insert_prediction(
            input_type="url" if url else ("title" if title else "text"),
            url=url,
            title=title,
            text=input_text,
            model_fake=model_scores["fake"],
            model_true=model_scores["true"],
            google_score=g_score,
            explanation=explanation,
        )
    except Exception:
        pass

    return model_scores, cred, google_items, g_score, explanation


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📰", layout="wide")
    st.title(APP_TITLE)

    tabs = st.tabs(["Predict", "Dashboard"])

    with tabs[0]:
        st.subheader("Analyze an article or claim")
        url = st.text_input("Article URL (optional)")
        title = st.text_input("Title (optional)")
        text = st.text_area("Article text or claim", height=180)

        if url and not text:
            if st.button("Fetch text from URL"):
                extracted = extract_text_from_url(url)
                if extracted:
                    st.session_state["auto_text"] = extracted
                    st.success("Extracted text from URL. Scroll to review.")
                else:
                    st.warning("Could not extract meaningful text from the URL.")

        if st.session_state.get("auto_text") and not text:
            text = st.session_state["auto_text"]
            st.text_area("Extracted text", value=text, height=240, key="extracted_text", disabled=True)

        col1, col2 = st.columns(2)
        with col1:
            run = st.button("Run Analysis", type="primary")
        with col2:
            clear = st.button("Clear")

        if clear:
            st.session_state.pop("auto_text", None)

        if run:
            if not (text or title or url):
                st.error("Please provide a URL, title, or text to analyze.")
            else:
                base_text = text or title or url
                with st.spinner("Running model, checking sources, asking LLM..."):
                    model_scores, cred, google_items, g_score, explanation = _predict_flow(
                        base_text, url=url or None, title=title or None
                    )

                st.markdown("### Results")
                st.metric("Model Credibility (0-100)", f"{cred:.1f}")
                st.progress(min(max(model_scores["true"], 0.0), 1.0))
                st.caption(f"Prob True: {model_scores['true']:.3f} | Prob Fake: {model_scores['fake']:.3f}")

                st.markdown("### Google Fact Check")
                if g_score is not None:
                    st.metric("Aggregated Fact Check Score (0-1)", f"{g_score:.2f}")
                else:
                    st.write("No fact-check results found or API not configured.")

                if google_items:
                    for it in google_items:
                        st.write(
                            f"- {it.get('textual_rating')} — {it.get('publisher')} — "
                            f"[source]({it.get('url')})"
                        )

                st.markdown("### Explanation (LLM)")
                if explanation:
                    st.write(explanation)
                else:
                    st.write("(OpenAI key not configured — skipping explanation.)")

    with tabs[1]:
        st.subheader("Recent Analyses")
        rows = fetch_recent(limit=200)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            try:
                import pandas as pd

                df = pd.DataFrame(rows)
                st.markdown("#### Score Over Time")
                st.line_chart(df.set_index("ts")["model_true"], height=220)
            except Exception:
                pass
        else:
            st.info("No results yet. Run an analysis first.")


if __name__ == "__main__":
    init_db()
    main()
