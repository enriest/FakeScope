import os
import logging
from typing import Optional, Dict, Any, List

import streamlit as st

from src.inference import credibility_score, predict_proba
from src.factcheck import fetch_fact_checks, aggregate_google_score, is_configured as gc_is_configured
from src.translate import translate_to_english
from src.openai_explain import generate_explanation
from src.storage import init_db, insert_prediction, fetch_recent
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
        "cost": "~$0.01-0.03 per analysis"
    },
    "gemini": {
        "name": "Google Gemini (1.5 Flash)",
        "description": "Best for high-volume usage and cost savings. Free tier with 1,500 requests/day. Fast and natural language understanding.",
        "strengths": "• FREE tier available\n• Very fast responses\n• Natural, conversational tone\n• Multimodal capable",
        "cost": "FREE (up to 1,500/day) or ~$0.005-0.01"
    },
    "perplexity": {
        "name": "Perplexity (Sonar)",
        "description": "Best for current events and recent news. Includes real-time web search, providing up-to-date context and additional sources.",
        "strengths": "• Real-time web search\n• Latest information\n• Automatically cites sources\n• Great for breaking news",
        "cost": "~$0.01-0.05 per analysis"
    }
}


def _predict_flow(model_text: str, query_text: str, url: Optional[str], title: Optional[str], language: str, llm_provider: str = None):
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


def _chat_with_llm(user_message: str, context: Dict[str, Any], llm_provider: str, chat_history: List[Dict]) -> str:
    """Generate a chat response from the LLM."""
    from src.openai_explain import _build_openai_client, _build_perplexity_client, _build_gemini_client, OPENAI_MODEL, PERPLEXITY_MODEL, GEMINI_MODEL
    import openai
    
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
                return "Gemini API not configured."
            
            model = client.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config={"temperature": 0.7, "max_output_tokens": 800}
            )
            
            # Build full conversation
            conversation = context_msg + "\n\nConversation:\n"
            for msg in chat_history[-6:]:  # Last 3 exchanges
                conversation += f"{msg['role'].title()}: {msg['content']}\n"
            conversation += f"User: {user_message}\nAssistant:"
            
            response = model.generate_content(conversation)
            return response.text.strip()
            
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
            {"role": "system", "content": "You are a helpful fact-checking assistant. Engage in thoughtful debate about the credibility of claims. Be open to the user's perspective but provide evidence-based counterarguments when appropriate."},
            {"role": "system", "content": context_msg}
        ]
        
        # Add chat history
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_message})
        
        resp = client.chat.completions.create(
            model=model,
            temperature=0.7,
            max_tokens=800,
            messages=messages
        )
        return resp.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📰", layout="wide")
    st.title(APP_TITLE)

    # Initialize session state
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None

    tabs = st.tabs(["🔍 Analyze", "💬 Chat & Debate", "⚖️ Compare Models", "📊 Deep Analysis", "📈 Dashboard"])

    # ============================================================
    # TAB 1: ANALYZE
    # ============================================================
    with tabs[0]:
        st.subheader("Analyze an article or claim")
        
        # LLM Provider Selection
        st.markdown("#### Choose your AI Model")
        selected_provider = st.selectbox(
            "LLM Provider",
            options=list(LLM_PROVIDERS.keys()),
            format_func=lambda x: LLM_PROVIDERS[x]["name"],
            key="primary_provider"
        )
        
        # Show provider info in expandable section
        with st.expander(f"ℹ️ Why {LLM_PROVIDERS[selected_provider]['name']}?", expanded=False):
            st.markdown(f"**{LLM_PROVIDERS[selected_provider]['description']}**")
            st.markdown("**Strengths:**")
            st.markdown(LLM_PROVIDERS[selected_provider]['strengths'])
            st.caption(f"💰 Cost: {LLM_PROVIDERS[selected_provider]['cost']}")
        
        url = st.text_input("Article URL (optional)")
        title = st.text_input("Title (optional)")
        text = st.text_area("Article text or claim", height=180)

        # Language selection for Google Fact Check API
        lang_options = ["en","es","fr","de","it","pt","ru","ar","zh","hi"]
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Fact Check Language", options=lang_options, index=0)
        with col2:
            auto_translate = st.checkbox("Auto-translate to English", value=True)

        if url and not text:
            if st.button("Fetch text from URL"):
                extracted = extract_text_from_url(url)
                if extracted:
                    st.session_state["auto_text"] = extracted
                    st.success("✅ Extracted text from URL")
                else:
                    st.warning("⚠️ Could not extract text")

        if st.session_state.get("auto_text") and not text:
            text = st.session_state["auto_text"]
            st.text_area("Extracted text", value=text, height=240, key="extracted_text", disabled=True)

        col1, col2 = st.columns(2)
        with col1:
            run = st.button("🚀 Run Analysis", type="primary", width="stretch")
        with col2:
            clear = st.button("🗑️ Clear", width="stretch")

        if clear:
            st.session_state.pop("auto_text", None)
            st.session_state.analysis_results = None
            st.session_state.chat_history = []
            st.rerun()

        if run:
            if not (text or title or url):
                st.error("⚠️ Please provide a URL, title, or text to analyze.")
            else:
                base_text = text or title or url
                with st.spinner(f"🔄 Analyzing with {LLM_PROVIDERS[selected_provider]['name']}..."):
                    model_input = base_text
                    translated_used = False
                    
                    if language != "en" and auto_translate:
                        translated = translate_to_english(base_text, language)
                        if translated and translated != base_text:
                            model_input = translated
                            translated_used = True
                    
                    model_scores, cred, google_items, g_score, explanation = _predict_flow(
                        model_input, base_text, url=url or None, title=title or None, 
                        language=language, llm_provider=selected_provider
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
                        "verdict": "TRUE" if cred >= 50 else "FAKE",
                        "translated": translated_used,
                        "language": language
                    }

        # Display results if available
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            # Verdict badge
            verdict_color = "🟢" if results["cred"] >= 70 else "🟡" if results["cred"] >= 40 else "🔴"
            st.markdown(f"## {verdict_color} Verdict: **{results['verdict']}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Credibility Score", f"{results['cred']:.1f}/100")
            with col2:
                st.metric("Fake Probability", f"{results['model_scores']['fake']:.1%}")
            with col3:
                st.metric("True Probability", f"{results['model_scores']['true']:.1%}")
            
            st.progress(min(max(results["model_scores"]["true"], 0.0), 1.0))
            
            if results.get("translated"):
                st.info(f"✅ Text was translated from {results['language'].upper()} to English for analysis")

            # Google Fact Check
            st.markdown("### 🌐 External Fact Checks")
            if results["g_score"] is not None:
                st.metric("Google Fact Check Score", f"{results['g_score']:.2f}/1.0")
            
            if results["google_items"]:
                for it in results["google_items"]:
                    rating = it.get('textual_rating', 'Unknown')
                    rating_icon = "✅" if "true" in rating.lower() else "❌" if "false" in rating.lower() else "⚠️"
                    st.markdown(f"{rating_icon} **{rating}** — {it.get('publisher')} — [View Source]({it.get('url')})")
            else:
                st.info("No external fact-checks found for this claim")

            # LLM Explanation
            st.markdown(f"### 🤖 Analysis by {LLM_PROVIDERS[results['provider']]['name']}")
            if results["explanation"]:
                st.markdown(results["explanation"])
            else:
                st.warning("LLM explanation not available. Check API configuration.")

    # ============================================================
    # TAB 2: CHAT & DEBATE
    # ============================================================
    with tabs[1]:
        st.subheader("💬 Debate with the AI")
        
        if not st.session_state.analysis_results:
            st.info("👈 Run an analysis first to start chatting!")
        else:
            results = st.session_state.analysis_results
            
            st.markdown(f"**Discussing:** {results['original_text'][:200]}...")
            st.caption(f"Using: {LLM_PROVIDERS[results['provider']]['name']}")
            
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
            user_input = st.text_area("Your argument or question:", height=100, key="chat_input")
            
            col1, col2 = st.columns([1, 5])
            with col1:
                send = st.button("Send", type="primary")
            with col2:
                clear_chat = st.button("Clear Chat")
            
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
                        "google_score": results["g_score"]
                    }
                    
                    # Get AI response
                    ai_response = _chat_with_llm(
                        user_input, 
                        context, 
                        results["provider"], 
                        st.session_state.chat_history
                    )
                    
                    # Add to history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    
                    st.rerun()
            
            # Quick prompts
            st.markdown("#### 💡 Quick Prompts")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Why is this fake/true?"):
                    st.session_state.chat_history.append({"role": "user", "content": "Explain in detail why you think this claim is fake or true."})
                    st.rerun()
            with col2:
                if st.button("Show me evidence"):
                    st.session_state.chat_history.append({"role": "user", "content": "What specific evidence supports or contradicts this claim?"})
                    st.rerun()
            with col3:
                if st.button("I disagree"):
                    st.session_state.chat_history.append({"role": "user", "content": "I disagree with your assessment. Can you consider alternative perspectives?"})
                    st.rerun()

    # ============================================================
    # TAB 3: COMPARE MODELS
    # ============================================================
    with tabs[2]:
        st.subheader("⚖️ Compare Different AI Models")
        
        if not st.session_state.analysis_results:
            st.info("👈 Run an analysis first to compare models!")
        else:
            results = st.session_state.analysis_results
            
            st.markdown(f"**Original Analysis:** {LLM_PROVIDERS[results['provider']]['name']}")
            
            # Select second provider
            available_providers = [p for p in LLM_PROVIDERS.keys() if p != results['provider']]
            second_provider = st.selectbox(
                "Compare with:",
                options=available_providers,
                format_func=lambda x: LLM_PROVIDERS[x]["name"]
            )
            
            if st.button("🔄 Run Comparison", type="primary"):
                with st.spinner(f"Running analysis with {LLM_PROVIDERS[second_provider]['name']}..."):
                    _, _, _, _, second_explanation = _predict_flow(
                        results["text"], results["original_text"], 
                        url=None, title=None, language=results["language"],
                        llm_provider=second_provider
                    )
                    
                    st.session_state.comparison_results = {
                        "provider": second_provider,
                        "explanation": second_explanation
                    }
            
            # Display comparison
            if st.session_state.comparison_results:
                comp = st.session_state.comparison_results
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {LLM_PROVIDERS[results['provider']]['name']}")
                    st.info(LLM_PROVIDERS[results['provider']]['description'])
                    st.markdown("**Analysis:**")
                    st.markdown(results["explanation"])
                
                with col2:
                    st.markdown(f"### {LLM_PROVIDERS[comp['provider']]['name']}")
                    st.info(LLM_PROVIDERS[comp['provider']]['description'])
                    st.markdown("**Analysis:**")
                    st.markdown(comp["explanation"])
                
                st.markdown("---")
                st.markdown("### 🎯 Key Differences")
                st.markdown("""
                Compare how each model approaches the analysis:
                - **Depth of explanation**: Which provides more detail?
                - **Source usage**: Which cites more external sources?
                - **Confidence level**: Which is more certain in its verdict?
                - **Writing style**: Which is clearer or more professional?
                """)

    # ============================================================
    # TAB 4: DEEP ANALYSIS
    # ============================================================
    with tabs[3]:
        st.subheader("📊 Deep Analysis & Sources")
        
        if not st.session_state.analysis_results:
            st.info("👈 Run an analysis first to see deep insights!")
        else:
            results = st.session_state.analysis_results
            
            # Source Analysis
            st.markdown("### 📰 Source Analysis")
            if results["google_items"]:
                st.markdown(f"**Found {len(results['google_items'])} external fact-checks**")
                
                # Count ratings
                ratings = {}
                for item in results["google_items"]:
                    rating = item.get('textual_rating', 'Unknown')
                    ratings[rating] = ratings.get(rating, 0) + 1
                
                # Display rating breakdown
                st.markdown("#### Rating Breakdown")
                for rating, count in ratings.items():
                    percentage = (count / len(results["google_items"])) * 100
                    st.progress(percentage / 100)
                    st.caption(f"{rating}: {count} sources ({percentage:.1f}%)")
                
                # Detailed source list
                st.markdown("#### Detailed Sources")
                for idx, item in enumerate(results["google_items"], 1):
                    with st.expander(f"Source {idx}: {item.get('publisher', 'Unknown')}"):
                        st.markdown(f"**Rating:** {item.get('textual_rating', 'Unknown')}")
                        st.markdown(f"**Claim:** {item.get('claim_text', 'N/A')}")
                        st.markdown(f"**URL:** [{item.get('url', 'N/A')}]({item.get('url', '#')})")
                        st.markdown(f"**Review Date:** {item.get('review_date', 'Unknown')}")
            else:
                st.warning("No external sources found. This claim may be too recent or too specific.")
            
            # Model Confidence Analysis
            st.markdown("### 🎯 Model Confidence")
            import pandas as pd
            confidence_data = pd.DataFrame({
                "Category": ["Fake", "True"],
                "Probability": [results["model_scores"]["fake"], results["model_scores"]["true"]]
            })
            st.bar_chart(confidence_data.set_index("Category"))
            
            # Text Statistics
            st.markdown("### 📈 Text Statistics")
            text = results["text"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Characters", len(text))
            with col2:
                st.metric("Words", len(text.split()))
            with col3:
                st.metric("Sentences", text.count('.') + text.count('!') + text.count('?'))
            with col4:
                avg_word_len = sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
                st.metric("Avg Word Length", f"{avg_word_len:.1f}")
            
            # Related Topics/Keywords
            st.markdown("### 🏷️ Key Topics")
            # Extract potential topics (simple approach)
            words = text.lower().split()
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            keywords = [w for w in words if len(w) > 5 and w not in common_words]
            if keywords:
                # Get unique keywords and show first 10
                unique_keywords = list(dict.fromkeys(keywords))[:10]
                st.write(", ".join([f"`{k}`" for k in unique_keywords]))
            else:
                st.info("No significant keywords identified")

    # ============================================================
    # TAB 5: DASHBOARD
    # ============================================================
    with tabs[4]:
        st.subheader("📈 Recent Analyses")
        rows = fetch_recent(limit=200)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            try:
                import pandas as pd
                df = pd.DataFrame(rows)
                st.markdown("#### Credibility Score Over Time")
                st.line_chart(df.set_index("ts")["model_true"], height=220)
            except Exception:
                pass
        else:
            st.info("No results yet. Run an analysis first.")


if __name__ == "__main__":
    init_db()
    main()
