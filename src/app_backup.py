import logging
import os
from typing import Any, Dict, Optional

import streamlit as st

from src.factcheck import aggregate_google_score, fetch_fact_checks
from src.factcheck import is_configured as gc_is_configured
from src.inference import credibility_score, predict_proba
from src.openai_explain import generate_chat_response, generate_explanation
from src.storage import fetch_recent, init_db, insert_prediction
from src.translate import translate_to_english
from src.utils import extract_text_from_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_TITLE = "FakeScope – Fake News Detector"

# LLM Provider descriptions
LLM_PROVIDERS = {
    "openai": {
        "name": "OpenAI (GPT-4o-mini)",
        "description": "Best for structured, reliable analysis. Excels at following instructions and generating well-formatted explanations.",
        "strengths": "• Consistent quality\n• Fast responses (1-2s)\n• Excellent for professional fact-checking",
        "cost": "~$0.01-0.03 per analysis",
    },
    "gemini": {
        "name": "Google Gemini (1.5 Flash)",
        "description": "Best for high-volume usage and cost savings. Free tier with 1,500 requests/day. Fast and natural language understanding.",
        "strengths": "• FREE tier available\n• Very fast responses\n• Natural, conversational tone\n• Multimodal capable",
        "cost": "FREE (up to 1,500/day) or ~$0.005-0.01",
    },
    "perplexity": {
        "name": "Perplexity (Sonar)",
        "description": "Best for current events and recent news. Includes real-time web search, providing up-to-date context and additional sources.",
        "strengths": "• Real-time web search\n• Latest information\n• Automatically cites sources\n• Great for breaking news",
        "cost": "~$0.01-0.05 per analysis",
    },
}


def _predict_flow(
    model_text: str,
    query_text: str,
    url: Optional[str],
    title: Optional[str],
    language: str,
    llm_provider: str = None,
):
    model_scores = predict_proba(model_text)
    cred = model_scores["true"] * 100.0

    google_items = fetch_fact_checks(title or query_text, language_code=language)
    g_score = aggregate_google_score(google_items)

    explanation = generate_explanation(
        input_text=model_text,
        model_scores=model_scores,
        google_items=google_items,
        google_score=g_score,
        provider_override=llm_provider,
    )

    # Persist
    try:
        insert_prediction(
            input_type="url" if url else ("title" if title else "text"),
            url=url,
            title=title,
            text=model_text,
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

        # Language selection for Google Fact Check API
        lang_options = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "hi"]
        language = st.selectbox(
            "Fact Check Language",
            options=lang_options,
            index=0,
            help="Language code passed to Google Fact Check Tools API.",
        )
        auto_translate = st.checkbox(
            "Auto-translate non-English to English",
            value=True,
            help="Translate claim/article before model & explanation when language != en.",
        )

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
            st.text_area(
                "Extracted text",
                value=text,
                height=240,
                key="extracted_text",
                disabled=True,
            )

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
                    model_input = base_text
                    translated_used = False
                    logger.info(
                        f"Translation check: language={language}, auto_translate={auto_translate}"
                    )
                    if language != "en" and auto_translate:
                        logger.info(f"Attempting translation from {language}")
                        logger.info(f"Original text preview: {base_text[:100]}...")
                        translated = translate_to_english(base_text, language)
                        logger.info(f"Translated text preview: {translated[:100]}...")
                        logger.info(
                            f"Translation result - original length: {len(base_text)}, translated length: {len(translated)}, different: {translated != base_text}"
                        )
                        # Use translated text even if similar, unless completely identical
                        if translated and len(translated.strip()) > 0:
                            if translated != base_text:
                                model_input = translated
                                translated_used = True
                                logger.info("Translation applied successfully")
                            else:
                                # Text unchanged - likely already English or mostly proper nouns
                                logger.warning(
                                    "Translation returned identical text - possibly already English or contains mostly proper nouns"
                                )
                                model_input = translated  # Use it anyway
                                translated_used = (
                                    False  # But mark as not translated for user info
                                )
                        else:
                            logger.warning("Translation returned empty or invalid")
                    model_scores, cred, google_items, g_score, explanation = (
                        _predict_flow(
                            model_input,
                            base_text,
                            url=url or None,
                            title=title or None,
                            language=language,
                        )
                    )

                st.markdown("### Results")
                st.metric("Model Credibility (0-100)", f"{cred:.1f}")
                st.progress(min(max(model_scores["true"], 0.0), 1.0))
                st.caption(
                    f"Prob True: {model_scores['true']:.3f} | Prob Fake: {model_scores['fake']:.3f}"
                )
                if language != "en" and auto_translate:
                    if translated_used:
                        st.caption(
                            "✅ Model & explanation ran on translated English text."
                        )
                    else:
                        st.caption(
                            "⚠️ Translation returned unchanged text (possibly already English or contains mostly proper nouns). Model used as-is."
                        )
                elif language != "en":
                    st.caption(
                        "ℹ️ Auto-translate is disabled. Model used original non-English text."
                    )

                st.markdown("### Google Fact Check")
                if g_score is not None:
                    st.metric("Aggregated Fact Check Score (0-1)", f"{g_score:.2f}")
                else:
                    if not gc_is_configured():
                        st.write("Google Fact Check API key not configured.")
                    else:
                        st.write("No fact-check results found for this query.")

                if google_items:
                    for it in google_items:
                        st.write(
                            f"- {it.get('textual_rating')} — {it.get('publisher')} — "
                            f"[source]({it.get('url')})"
                        )

                # Diagnostics line
                api_status = "yes" if gc_is_configured() else "no"
                st.caption(f"API configured: {api_status} | Query language: {language}")

                st.markdown("### Explanation (LLM)")
                if explanation:
                    st.write(explanation)
                else:
                    st.write(
                        "(LLM API key not configured — skipping explanation. Set OPENAI_API_KEY, PERPLEXITY_API_KEY, or GEMINI_API_KEY and FAKESCOPE_LLM_PROVIDER.)"
                    )

                # Show translation if it was applied
                if language != "en" and auto_translate:
                    if translated_used and model_input != base_text:
                        with st.expander("📝 View Translation"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Original ({language.upper()}):**")
                                st.text_area(
                                    "",
                                    base_text,
                                    height=150,
                                    key="orig_text",
                                    disabled=True,
                                    label_visibility="collapsed",
                                )
                            with col2:
                                st.markdown("**Translated (EN):**")
                                st.text_area(
                                    "",
                                    model_input,
                                    height=150,
                                    key="trans_text",
                                    disabled=True,
                                    label_visibility="collapsed",
                                )
                    elif not translated_used:
                        with st.expander("ℹ️ Translation Note"):
                            st.info(
                                "Translation API returned the text unchanged. This typically means the text already contains mostly English words, proper nouns, or technical terms that don't require translation."
                            )
                            st.text_area(
                                "Original Text:", base_text, height=100, disabled=True
                            )

    with tabs[1]:
        st.subheader("Recent Analyses")
        rows = fetch_recent(limit=200)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
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
