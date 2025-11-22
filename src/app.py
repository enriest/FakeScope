import logging
import os
from typing import Any, Dict, List, Optional

import streamlit as st

# Load .env so the app picks up API keys without requiring `source .env`
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from src.factcheck import aggregate_google_score, fetch_fact_checks
from src.factcheck import is_configured as gc_is_configured
from src.i18n import (SUPPORTED_LANGUAGES, detect_language_from_ip,
                      get_language_name, get_translation)
from src.inference import credibility_score, predict_proba
from src.openai_explain import generate_explanation
from src.storage import fetch_recent, init_db, insert_prediction
from src.translate import translate_to_english
from src.utils import extract_text_from_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Run prediction flow with specified LLM provider."""
    model_scores = predict_proba(model_text)
    cred = model_scores["true"] * 100.0

    google_items = fetch_fact_checks(title or query_text, language_code=language)
    g_score = aggregate_google_score(google_items)

    # Override provider if specified
    original_provider = os.getenv("FAKESCOPE_LLM_PROVIDER", "openai")
    if llm_provider:
        os.environ["FAKESCOPE_LLM_PROVIDER"] = llm_provider

    explanation = generate_explanation(
        input_text=model_text,
        model_scores=model_scores,
        google_items=google_items,
        google_score=g_score,
    )

    # Restore original provider
    os.environ["FAKESCOPE_LLM_PROVIDER"] = original_provider

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


def _chat_with_llm(
    user_message: str,
    context: Dict[str, Any],
    llm_provider: str,
    chat_history: List[Dict],
) -> str:
    """Generate a chat response from the LLM."""
    import openai

    from src.openai_explain import (GEMINI_MODEL, OPENAI_MODEL,
                                    PERPLEXITY_MODEL, _build_gemini_client,
                                    _build_openai_client,
                                    _build_perplexity_client)

    provider = llm_provider.lower()

    # Build context message
    context_msg = f"""Previous Analysis Context:
- Claim: {context.get('text', '')[:500]}...
- Model Verdict: {context.get('verdict', 'Unknown')} (Credibility: {context.get('cred', 0):.1f}/100)
- Prob Fake: {context.get('fake_prob', 0):.3f}, Prob True: {context.get('true_prob', 0):.3f}
- Google Fact Check Score: {context.get('google_score', 'N/A')}

User wants to discuss: {user_message}
"""

    try:
        if provider == "gemini":
            client = _build_gemini_client()
            if not client:
                return "Gemini API not configured. Set GEMINI_API_KEY environment variable."

            try:
                model = client.GenerativeModel(
                    model_name=GEMINI_MODEL,
                    generation_config={"temperature": 0.7, "max_output_tokens": 800},
                )

                # Build full conversation
                conversation = context_msg + "\n\nConversation:\n"
                for msg in chat_history[-6:]:  # Last 3 exchanges
                    conversation += f"{msg['role'].title()}: {msg['content']}\n"
                conversation += f"User: {user_message}\nAssistant:"

                response = model.generate_content(conversation)
                # Safely access text attribute (can raise ValueError if blocked/empty)
                try:
                    text = response.text.strip()
                    if text:
                        return text
                    return "(Gemini returned empty response)"
                except (ValueError, AttributeError) as e:
                    return f"(Gemini response unavailable: {str(e)})"
            except Exception as e:
                return f"Gemini API error: {str(e)}"

        elif provider == "perplexity":
            client = _build_perplexity_client()
            model = PERPLEXITY_MODEL
        else:
            client = _build_openai_client()
            model = OPENAI_MODEL

        if not client:
            return f"{provider.title()} API not configured."

        # Build messages for OpenAI-compatible APIs
        messages = [
            {
                "role": "system",
                "content": "You are a helpful fact-checking assistant. Engage in thoughtful debate about the credibility of claims. Be open to the user's perspective but provide evidence-based counterarguments when appropriate.",
            },
            {"role": "system", "content": context_msg},
        ]

        # Add chat history
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        resp = client.chat.completions.create(
            model=model, temperature=0.7, max_tokens=800, messages=messages
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    # Initialize session state early
    if "ui_language" not in st.session_state:
        # Try to detect language from IP on first load
        try:
            # Get client IP (Streamlit Cloud provides this via headers)
            # For local development, this will return None and default to English
            client_ip = (
                st.context.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            )
            if not client_ip:
                client_ip = st.context.headers.get("X-Real-Ip", "")
            detected_lang = detect_language_from_ip(client_ip) if client_ip else "en"
            st.session_state.ui_language = detected_lang
        except Exception:
            st.session_state.ui_language = "en"

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None

    # Get current language
    current_lang = st.session_state.ui_language
    t = lambda key, **kwargs: get_translation(key, current_lang, **kwargs)

    # Page config and title
    st.set_page_config(page_title=t("app_title"), page_icon="📰", layout="wide")

    # Title and language selector in same row
    col_title, col_lang = st.columns([4, 1])
    with col_title:
        st.title(t("app_title"))
    with col_lang:
        # Language selector
        lang_display = {code: get_language_name(code) for code in SUPPORTED_LANGUAGES}
        selected_lang = st.selectbox(
            t("language"),
            options=SUPPORTED_LANGUAGES,
            format_func=lambda x: lang_display[x],
            index=SUPPORTED_LANGUAGES.index(current_lang),
            key="language_selector",
        )

        # Update language if changed
        if selected_lang != current_lang:
            st.session_state.ui_language = selected_lang
            st.rerun()

    tabs = st.tabs(
        [
            t("analyze_tab"),
            t("chat_tab"),
            t("compare_tab"),
            t("deep_analysis_tab"),
            t("dashboard_tab"),
        ]
    )

    # ============================================================
    # TAB 1: ANALYZE
    # ============================================================
    with tabs[0]:
        st.subheader(t("analyze_subtitle"))

        # LLM Provider Selection
        st.markdown(f"#### {t('choose_ai_model')}")
        selected_provider = st.selectbox(
            t("llm_provider"),
            options=list(LLM_PROVIDERS.keys()),
            format_func=lambda x: LLM_PROVIDERS[x]["name"],
            key="primary_provider",
        )

        # If provider changed, invalidate previous analysis to avoid showing stale explanation
        if (
            st.session_state.get("analysis_results")
            and st.session_state.analysis_results.get("provider") != selected_provider
        ):
            st.session_state.analysis_results = None
            st.info(
                t("provider_changed_run_again")
                if callable(t)
                else "Provider changed. Click Run Analysis to generate a fresh explanation."
            )

        # Show provider info in expandable section
        with st.expander(
            f"ℹ️ {t('why_provider')} {LLM_PROVIDERS[selected_provider]['name']}?",
            expanded=False,
        ):
            # Provider description and strengths via i18n
            st.markdown(
                f"**{get_translation(f'provider_{selected_provider}_description', current_lang)}**"
            )
            st.markdown(f"**{t('strengths')}**")
            st.markdown(
                get_translation(f"provider_{selected_provider}_strengths", current_lang)
            )
            st.caption(f"💰 {t('cost')} {LLM_PROVIDERS[selected_provider]['cost']}")

        url = st.text_input(t("article_url"))
        title = st.text_input(t("title_optional"))
        text = st.text_area(t("article_text"), height=180)

        # Language selection for Google Fact Check API
        lang_options = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "hi"]
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox(
                t("fact_check_language"), options=lang_options, index=0
            )
        with col2:
            auto_translate = st.checkbox(t("auto_translate"), value=True)

        if url and not text:
            if st.button(t("fetch_from_url")):
                extracted = extract_text_from_url(url)
                if extracted:
                    st.session_state["auto_text"] = extracted
                    st.success(t("extracted_success"))
                else:
                    st.warning(t("extract_failed"))

        if st.session_state.get("auto_text") and not text:
            text = st.session_state["auto_text"]
            st.text_area(
                t("extracted_text"),
                value=text,
                height=240,
                key="extracted_text",
                disabled=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            run = st.button(t("run_analysis"), type="primary", width="stretch")
        with col2:
            clear = st.button(t("clear"), width="stretch")

        if clear:
            st.session_state.pop("auto_text", None)
            st.session_state.analysis_results = None
            st.session_state.chat_history = []
            st.rerun()

        if run:
            if not (text or title or url):
                st.error(t("provide_input_error"))
            else:
                base_text = text or title or url
                with st.spinner(
                    f"{t('analyzing')} {LLM_PROVIDERS[selected_provider]['name']}..."
                ):
                    model_input = base_text
                    translated_used = False

                    if language != "en" and auto_translate:
                        translated = translate_to_english(base_text, language)
                        if translated and translated != base_text:
                            model_input = translated
                            translated_used = True

                    model_scores, cred, google_items, g_score, explanation = (
                        _predict_flow(
                            model_input,
                            base_text,
                            url=url or None,
                            title=title or None,
                            language=language,
                            llm_provider=selected_provider,
                        )
                    )

                    # Store results in session state
                    st.session_state.analysis_results = {
                        "text": model_input,
                        "original_text": base_text,
                        "model_scores": model_scores,
                        "cred": cred,
                        "google_items": google_items,
                        "g_score": g_score,
                        "explanation": explanation,
                        "provider": selected_provider,
                        "verdict": t("true") if cred >= 50 else t("fake"),
                        "translated": translated_used,
                        "language": language,
                    }

        # Display results if available
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results

            st.markdown("---")
            st.markdown(f"### {t('results')}")

            # Verdict badge
            verdict_color = (
                "🟢"
                if results["cred"] >= 70
                else "🟡" if results["cred"] >= 40 else "🔴"
            )
            st.markdown(f"## {verdict_color} {t('verdict')} **{results['verdict']}**")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t("credibility_score"), f"{results['cred']:.1f}/100")
            with col2:
                st.metric(
                    t("fake_probability"), f"{results['model_scores']['fake']:.1%}"
                )
            with col3:
                st.metric(
                    t("true_probability"), f"{results['model_scores']['true']:.1%}"
                )

            st.progress(min(max(results["model_scores"]["true"], 0.0), 1.0))

            if results.get("translated"):
                st.info(t("translated_info", lang=results["language"].upper()))

            # Google Fact Check
            st.markdown(f"### {t('external_fact_checks')}")
            if results["g_score"] is not None:
                st.metric(t("google_fact_check_score"), f"{results['g_score']:.2f}/1.0")

            if results["google_items"]:
                for it in results["google_items"]:
                    rating = it.get("textual_rating", "Unknown")
                    rating_icon = (
                        "✅"
                        if "true" in rating.lower()
                        else "❌" if "false" in rating.lower() else "⚠️"
                    )
                    st.markdown(
                        f"{rating_icon} **{rating}** — {it.get('publisher')} — [{t('view_source')}]({it.get('url')})"
                    )
            else:
                st.info(t("no_fact_checks"))

            # LLM Explanation
            st.markdown(f"### {t('llm_explanation')}")
            st.caption(
                f"{t('provided_by')} {LLM_PROVIDERS[results['provider']]['name']}"
            )
            if results["explanation"]:
                # Optional debug info if env var set
                if os.getenv("DEBUG_UI_EXPLANATION"):
                    st.caption(
                        f"[debug] provider={results['provider']} chars={len(results['explanation'])}"
                    )
                st.markdown(results["explanation"])
            else:
                st.warning("LLM explanation not available. Check API configuration.")

    # ============================================================
    # TAB 2: CHAT & DEBATE
    # ============================================================
    with tabs[1]:
        st.subheader(t("chat_subtitle"))

        if not st.session_state.analysis_results:
            st.info(t("analyze_first"))
        else:
            results = st.session_state.analysis_results

            st.markdown(f"**{t('current_verdict')}:** {results['verdict']}")
            st.caption(
                f"{t('provided_by')} {LLM_PROVIDERS[results['provider']]['name']}"
            )

            # Display chat history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"**You:** {msg['content']}")
                    else:
                        st.markdown(f"**AI:** {msg['content']}")
                    st.markdown("---")

            # Chat input
            user_input = st.text_area(t("your_message"), height=100, key="chat_input")

            col1, col2 = st.columns([1, 5])
            with col1:
                send = st.button(t("send"), type="primary")
            with col2:
                clear_chat = st.button(t("clear"))

            if clear_chat:
                st.session_state.chat_history = []
                st.rerun()

            if send and user_input:
                with st.spinner("🤔 AI is thinking..."):
                    # Prepare context
                    context = {
                        "text": results["text"],
                        "verdict": results["verdict"],
                        "cred": results["cred"],
                        "fake_prob": results["model_scores"]["fake"],
                        "true_prob": results["model_scores"]["true"],
                        "google_score": results["g_score"],
                    }

                    # Get AI response
                    ai_response = _chat_with_llm(
                        user_input,
                        context,
                        results["provider"],
                        st.session_state.chat_history,
                    )

                    # Add to history
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_input}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": ai_response}
                    )

                    st.rerun()

            # Quick prompts
            st.markdown(f"#### {t('quick_prompts_title')}")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(t("quick_prompt_why")):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": t("quick_prompt_msg_why")}
                    )
                st.rerun()
            with col2:
                if st.button(t("quick_prompt_evidence")):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": t("quick_prompt_msg_evidence")}
                    )
                    st.rerun()
            with col3:
                if st.button(t("quick_prompt_disagree")):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": t("quick_prompt_msg_disagree")}
                    )
                    st.rerun()

    # ============================================================
    # TAB 3: COMPARE MODELS
    # ============================================================
    with tabs[2]:
        st.subheader(t("compare_subtitle"))

        if not st.session_state.analysis_results:
            st.info(t("analyze_first"))
        else:
            results = st.session_state.analysis_results

            st.markdown(
                f"**{t('model')}:** {LLM_PROVIDERS[results['provider']]['name']}"
            )

            # Select second provider
            available_providers = [
                p for p in LLM_PROVIDERS.keys() if p != results["provider"]
            ]
            second_provider = st.selectbox(
                (
                    t("compare_with")
                    if hasattr(__import__("builtins"), "True")
                    else "Compare with:"
                ),
                options=available_providers,
                format_func=lambda x: LLM_PROVIDERS[x]["name"],
            )

            if st.button(t("compare_button"), type="primary"):
                with st.spinner(t("comparing")):
                    _, _, _, _, second_explanation = _predict_flow(
                        results["text"],
                        results["original_text"],
                        url=None,
                        title=None,
                        language=results["language"],
                        llm_provider=second_provider,
                    )

                    st.session_state.comparison_results = {
                        "provider": second_provider,
                        "explanation": second_explanation,
                    }

            # Display comparison
            if st.session_state.comparison_results:
                comp = st.session_state.comparison_results

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### {LLM_PROVIDERS[results['provider']]['name']}")
                    st.info(LLM_PROVIDERS[results["provider"]]["description"])
                    st.markdown("**Analysis:**")
                    st.markdown(results["explanation"])

                with col2:
                    st.markdown(f"### {LLM_PROVIDERS[comp['provider']]['name']}")
                    st.info(LLM_PROVIDERS[comp["provider"]]["description"])
                    st.markdown("**Analysis:**")
                    st.markdown(comp["explanation"])

                st.markdown("---")
                st.markdown("### 🎯 Key Differences")
                st.markdown(
                    """
                Compare how each model approaches the analysis:
                - **Depth of explanation**: Which provides more detail?
                - **Source usage**: Which cites more external sources?
                - **Confidence level**: Which is more certain in its verdict?
                - **Writing style**: Which is clearer or more professional?
                """
                )

    # ============================================================
    # TAB 4: DEEP ANALYSIS
    # ============================================================
    with tabs[3]:
        st.subheader(t("deep_analysis_subtitle"))

        if not st.session_state.analysis_results:
            st.info(t("analyze_first"))
        else:
            results = st.session_state.analysis_results

            # Source Analysis
            st.markdown(f"### {t('source_analysis')}")
            if results["google_items"]:
                st.markdown(
                    f"**{t('found_fact_checks_count', count=len(results['google_items']))}**"
                )

                # Count ratings
                ratings = {}
                for item in results["google_items"]:
                    rating = item.get("textual_rating", "Unknown")
                    ratings[rating] = ratings.get(rating, 0) + 1

                # Display rating breakdown
                st.markdown(f"#### {t('rating_breakdown')}")
                for rating, count in ratings.items():
                    percentage = (count / len(results["google_items"])) * 100
                    st.progress(percentage / 100)
                    st.caption(f"{rating}: {count} sources ({percentage:.1f}%)")

                # Detailed source list
                st.markdown(f"#### {t('detailed_sources')}")
                for idx, item in enumerate(results["google_items"], 1):
                    with st.expander(
                        t(
                            "source_item",
                            idx=idx,
                            publisher=item.get("publisher", "Unknown"),
                        )
                    ):
                        st.markdown(
                            f"**{t('rating_label')}:** {item.get('textual_rating', 'Unknown')}"
                        )
                        st.markdown(
                            f"**{t('claim_label')}:** {item.get('claim_text', 'N/A')}"
                        )
                        st.markdown(
                            f"**{t('url_label')}:** [{item.get('url', 'N/A')}]({item.get('url', '#')})"
                        )
                        st.markdown(
                            f"**{t('review_date_label')}:** {item.get('review_date', 'Unknown')}"
                        )
            else:
                st.warning(t("no_external_sources"))

            # Model Confidence Analysis
            st.markdown(f"### {t('model_confidence')}")
            import pandas as pd

            confidence_data = pd.DataFrame(
                {
                    "Category": [t("fake"), t("true")],
                    "Probability": [
                        results["model_scores"]["fake"],
                        results["model_scores"]["true"],
                    ],
                }
            )
            st.bar_chart(confidence_data.set_index("Category"))

            # Text Statistics
            st.markdown(f"### {t('text_statistics')}")
            text = results["text"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(t("characters"), len(text))
            with col2:
                st.metric(t("words"), len(text.split()))
            with col3:
                st.metric(
                    t("sentences"), text.count(".") + text.count("!") + text.count("?")
                )
            with col4:
                avg_word_len = (
                    sum(len(word) for word in text.split()) / len(text.split())
                    if text.split()
                    else 0
                )
                st.metric(t("avg_word_length"), f"{avg_word_len:.1f}")

            # Related Topics/Keywords
            st.markdown(f"### {t('key_topics')}")
            # Extract potential topics (simple approach)
            words = text.lower().split()
            common_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
            }
            keywords = [w for w in words if len(w) > 5 and w not in common_words]
            if keywords:
                # Get unique keywords and show first 10
                unique_keywords = list(dict.fromkeys(keywords))[:10]
                st.write(", ".join([f"`{k}`" for k in unique_keywords]))
            else:
                st.info(t("no_keywords_identified"))

    # ============================================================
    # TAB 5: DASHBOARD
    # ============================================================
    with tabs[4]:
        st.subheader(t("dashboard_subtitle"))
        rows = fetch_recent(limit=200)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            try:
                import pandas as pd

                df = pd.DataFrame(rows)
                st.markdown(f"#### {t('credibility_score')}")
                st.line_chart(df.set_index("ts")["model_true"], height=220)
            except Exception:
                pass
        else:
            st.info(t("no_data"))


if __name__ == "__main__":
    init_db()
    main()
