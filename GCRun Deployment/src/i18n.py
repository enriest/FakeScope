"""
Internationalization (i18n) module for FakeScope
Supports: English (default), Spanish, French, German, Russian, Portuguese
"""

import os
from typing import Optional

import requests

# Translation dictionary
TRANSLATIONS = {
    "en": {
        # App title and main UI
        "app_title": "FakeScope – Fake News Detector",
        "analyze_tab": "🔍 Analyze",
        "chat_tab": "💬 Chat & Debate",
        "compare_tab": "⚖️ Compare Models",
        "deep_analysis_tab": "📊 Deep Analysis",
        "dashboard_tab": "📈 Dashboard",
        # Language selector
        "language": "Language",
        # Analyze section
        "analyze_subtitle": "Analyze an article or claim",
        "choose_ai_model": "Choose your AI Model",
        "llm_provider": "LLM Provider",
        "why_provider": "Why",
        "strengths": "Strengths:",
        "cost": "Cost:",
        "article_url": "Article URL (optional)",
        "title_optional": "Title (optional)",
        "article_text": "Article text or claim",
        "fact_check_language": "Fact Check Language",
        "auto_translate": "Auto-translate to English",
        "fetch_from_url": "Fetch text from URL",
        "extracted_text": "Extracted text",
        "run_analysis": "🚀 Run Analysis",
        "clear": "🗑️ Clear",
        "provide_input_error": "⚠️ Please provide a URL, title, or text to analyze.",
        "analyzing": "🔄 Analyzing with",
        "extracted_success": "✅ Extracted text from URL",
        "extract_failed": "⚠️ Could not extract text",
        # Results section
        "results": "📊 Results",
        "verdict": "Verdict:",
        "true": "TRUE",
        "fake": "FAKE",
        "credibility_score": "Credibility Score",
        "fake_probability": "Fake Probability",
        "true_probability": "True Probability",
        "translated_info": "✅ Text was translated from {lang} to English for analysis",
        # External fact checks
        "external_fact_checks": "🌐 External Fact Checks",
        "google_fact_check_score": "Google Fact Check Score",
        "claim": "Claim",
        "rating": "Rating",
        "publisher": "Publisher",
        "review_date": "Review Date",
        "no_fact_checks": "No external fact checks found for this claim.",
        # LLM Explanation
        "llm_explanation": "🤖 AI Explanation",
        "provided_by": "Provided by",
        # Chat section
        "start_conversation": "Start a conversation about the analysis",
        "chat_subtitle": "Discuss and debate the credibility of your analyzed claim",
        "analyze_first": "👈 Please run an analysis first in the **Analyze** tab",
        "current_verdict": "Current Verdict",
        "your_message": "Your message",
        "send": "Send",
        "chat_history": "Chat History",
        # Compare models
        "compare_subtitle": "Compare results from all three AI models side-by-side",
        "input_to_compare": "Enter text to compare across models",
        "compare_button": "⚖️ Compare All Models",
        "comparing": "Comparing across OpenAI, Gemini, and Perplexity...",
        "model_comparison": "Model Comparison Results",
        "model": "Model",
        "response": "Response",
        "response_time": "Response Time",
        "seconds": "seconds",
        # Deep analysis
        "deep_analysis_subtitle": "Get comprehensive analysis with sources and related news",
        "deep_analysis_button": "🔍 Run Deep Analysis",
        "deep_analyzing": "Running deep analysis...",
        "sources_found": "Sources Found",
        "related_articles": "Related Articles",
        "sentiment_analysis": "Sentiment Analysis",
        "key_entities": "Key Entities",
        # Dashboard
        "dashboard_subtitle": "Recent analyses and statistics",
        "recent_analyses": "Recent Analyses",
        "no_data": "No data available yet. Run some analyses first!",
        "total_analyses": "Total Analyses",
        "avg_credibility": "Average Credibility",
        "most_used_provider": "Most Used Provider",
        # Provider details (names remain from LLM_PROVIDERS)
        "provider_openai_description": "Best for structured, reliable analysis. Excels at following instructions and generating well-formatted explanations.",
        "provider_openai_strengths": "• Consistent quality\n• Fast responses (1-2s)\n• Excellent for professional fact-checking",
        "provider_gemini_description": "Best for high-volume usage and cost savings. Free tier with 1,500 requests/day. Fast and natural language understanding.",
        "provider_gemini_strengths": "• FREE tier available\n• Very fast responses\n• Natural, conversational tone\n• Multimodal capable",
        "provider_perplexity_description": "Best for current events and recent news. Includes real-time web search, providing up-to-date context and additional sources.",
        "provider_perplexity_strengths": "• Real-time web search\n• Latest information\n• Automatically cites sources\n• Great for breaking news",
        # External links / labels
        "view_source": "View Source",
        # Chat quick prompts
        "quick_prompts_title": "💡 Quick Prompts",
        "quick_prompt_why": "Why is this fake/true?",
        "quick_prompt_evidence": "Show me evidence",
        "quick_prompt_disagree": "I disagree",
        "quick_prompt_msg_why": "Explain in detail why you think this claim is fake or true.",
        "quick_prompt_msg_evidence": "What specific evidence supports or contradicts this claim?",
        "quick_prompt_msg_disagree": "I disagree with your assessment. Can you consider alternative perspectives?",
        # Deep analysis section
        "source_analysis": "📰 Source Analysis",
        "found_fact_checks_count": "Found {count} external fact-checks",
        "rating_breakdown": "Rating Breakdown",
        "detailed_sources": "Detailed Sources",
        "source_item": "Source {idx}: {publisher}",
        "rating_label": "Rating",
        "claim_label": "Claim",
        "url_label": "URL",
        "review_date_label": "Review Date",
        "no_external_sources": "No external sources found. This claim may be too recent or too specific.",
        "model_confidence": "🎯 Model Confidence",
        "text_statistics": "📈 Text Statistics",
        "characters": "Characters",
        "words": "Words",
        "sentences": "Sentences",
        "avg_word_length": "Avg Word Length",
        "key_topics": "🏷️ Key Topics",
        "no_keywords_identified": "No significant keywords identified",
        "provider_changed_run_again": "LLM provider changed. Click 'Run Analysis' to generate a new explanation.",
        "key_differences_title": "🎯 Key Differences",
        "key_differences_body": """Compare how each model approaches the analysis:\n- **Depth of explanation**: Which provides more detail?\n- **Source usage**: Which cites more external sources?\n- **Confidence level**: Which is more certain in its verdict?\n- **Writing style**: Which is clearer or more professional?""",
        "gemini_api_missing": "Gemini API key not set. Add GEMINI_API_KEY in Space settings.",
        "gemini_api_failed": "Gemini request failed. Check key, model name, or quota.",
    },
    "es": {
        # App title and main UI
        "app_title": "FakeScope – Detector de Noticias Falsas",
        "analyze_tab": "🔍 Analizar",
        "chat_tab": "💬 Chat y Debate",
        "compare_tab": "⚖️ Comparar Modelos",
        "deep_analysis_tab": "📊 Análisis Profundo",
        "dashboard_tab": "📈 Panel",
        # Language selector
        "language": "Idioma",
        # Analyze section
        "analyze_subtitle": "Analizar un artículo o afirmación",
        "choose_ai_model": "Elige tu Modelo de IA",
        "llm_provider": "Proveedor LLM",
        "why_provider": "Por qué",
        "strengths": "Fortalezas:",
        "cost": "Costo:",
        "article_url": "URL del artículo (opcional)",
        "title_optional": "Título (opcional)",
        "article_text": "Texto del artículo o afirmación",
        "fact_check_language": "Idioma de Verificación",
        "auto_translate": "Traducir automáticamente al inglés",
        "fetch_from_url": "Obtener texto de la URL",
        "extracted_text": "Texto extraído",
        "run_analysis": "🚀 Ejecutar Análisis",
        "clear": "🗑️ Limpiar",
        "provide_input_error": "⚠️ Por favor proporciona una URL, título o texto para analizar.",
        "analyzing": "🔄 Analizando con",
        "extracted_success": "✅ Texto extraído de la URL",
        "extract_failed": "⚠️ No se pudo extraer el texto",
        # Results section
        "results": "📊 Resultados",
        "verdict": "Veredicto:",
        "true": "VERDADERO",
        "fake": "FALSO",
        "credibility_score": "Puntuación de Credibilidad",
        "fake_probability": "Probabilidad Falso",
        "true_probability": "Probabilidad Verdadero",
        "translated_info": "✅ El texto fue traducido de {lang} al inglés para el análisis",
        # External fact checks
        "external_fact_checks": "🌐 Verificaciones Externas",
        "google_fact_check_score": "Puntuación Google Fact Check",
        "claim": "Afirmación",
        "rating": "Calificación",
        "publisher": "Editor",
        "review_date": "Fecha de Revisión",
        "no_fact_checks": "No se encontraron verificaciones externas para esta afirmación.",
        # LLM Explanation
        "llm_explanation": "🤖 Explicación de IA",
        "provided_by": "Proporcionado por",
        # Chat section
        "start_conversation": "Iniciar una conversación sobre el análisis",
        "chat_subtitle": "Discutir y debatir la credibilidad de tu afirmación analizada",
        "analyze_first": "👈 Por favor ejecuta un análisis primero en la pestaña **Analizar**",
        "current_verdict": "Veredicto Actual",
        "your_message": "Tu mensaje",
        "send": "Enviar",
        "chat_history": "Historial de Chat",
        # Compare models
        "compare_subtitle": "Compara resultados de los tres modelos de IA lado a lado",
        "input_to_compare": "Ingresa texto para comparar entre modelos",
        "compare_button": "⚖️ Comparar Todos los Modelos",
        "comparing": "Comparando entre OpenAI, Gemini y Perplexity...",
        "model_comparison": "Resultados de Comparación de Modelos",
        "model": "Modelo",
        "response": "Respuesta",
        "response_time": "Tiempo de Respuesta",
        "seconds": "segundos",
        # Deep analysis
        "deep_analysis_subtitle": "Obtén análisis completo con fuentes y noticias relacionadas",
        "deep_analysis_button": "🔍 Ejecutar Análisis Profundo",
        "deep_analyzing": "Ejecutando análisis profundo...",
        "sources_found": "Fuentes Encontradas",
        "related_articles": "Artículos Relacionados",
        "sentiment_analysis": "Análisis de Sentimiento",
        "key_entities": "Entidades Clave",
        # Dashboard
        "dashboard_subtitle": "Análisis recientes y estadísticas",
        "recent_analyses": "Análisis Recientes",
        "no_data": "¡No hay datos disponibles aún. Ejecuta algunos análisis primero!",
        "total_analyses": "Total de Análisis",
        "avg_credibility": "Credibilidad Promedio",
        "most_used_provider": "Proveedor Más Usado",
        # Provider details
        "provider_openai_description": "Ideal para análisis estructurado y confiable. Destaca en seguir instrucciones y generar explicaciones bien formateadas.",
        "provider_openai_strengths": "• Calidad constante\n• Respuestas rápidas (1-2s)\n• Excelente para verificación profesional",
        "provider_gemini_description": "Ideal para alto volumen y ahorro de costos. Plan gratuito con 1.500 solicitudes/día. Rápido y con comprensión natural del lenguaje.",
        "provider_gemini_strengths": "• Plan GRATIS disponible\n• Respuestas muy rápidas\n• Tono natural y conversacional\n• Capaz de trabajar con múltiples modalidades",
        "provider_perplexity_description": "Ideal para eventos actuales y noticias recientes. Incluye búsqueda web en tiempo real, ofreciendo contexto actualizado y fuentes adicionales.",
        "provider_perplexity_strengths": "• Búsqueda web en tiempo real\n• Información más reciente\n• Cita fuentes automáticamente\n• Excelente para noticias de última hora",
        # External links / labels
        "view_source": "Ver Fuente",
        # Chat quick prompts
        "quick_prompts_title": "💡 Sugerencias Rápidas",
        "quick_prompt_why": "¿Por qué es falso/verdadero?",
        "quick_prompt_evidence": "Muéstrame evidencia",
        "quick_prompt_disagree": "No estoy de acuerdo",
        "quick_prompt_msg_why": "Explica en detalle por qué crees que esta afirmación es falsa o verdadera.",
        "quick_prompt_msg_evidence": "¿Qué evidencia específica apoya o contradice esta afirmación?",
        "quick_prompt_msg_disagree": "No estoy de acuerdo con tu evaluación. ¿Puedes considerar perspectivas alternativas?",
        # Deep analysis section
        "source_analysis": "📰 Análisis de Fuentes",
        "found_fact_checks_count": "Se encontraron {count} verificaciones externas",
        "rating_breakdown": "Desglose de Calificaciones",
        "detailed_sources": "Fuentes Detalladas",
        "source_item": "Fuente {idx}: {publisher}",
        "rating_label": "Calificación",
        "claim_label": "Afirmación",
        "url_label": "URL",
        "review_date_label": "Fecha de Revisión",
        "no_external_sources": "No se encontraron fuentes externas. Esta afirmación puede ser demasiado reciente o específica.",
        "model_confidence": "🎯 Confianza del Modelo",
        "text_statistics": "📈 Estadísticas del Texto",
        "characters": "Caracteres",
        "words": "Palabras",
        "sentences": "Oraciones",
        "avg_word_length": "Longitud Media de Palabra",
        "key_topics": "🏷️ Temas Clave",
        "no_keywords_identified": "No se identificaron palabras clave significativas",
        "provider_changed_run_again": "El proveedor de IA cambió. Pulsa 'Ejecutar Análisis' para generar una nueva explicación.",
        "key_differences_title": "🎯 Diferencias Clave",
        "key_differences_body": """Compara cómo cada modelo aborda el análisis:\n- **Profundidad de la explicación**: ¿Cuál aporta más detalle?\n- **Uso de fuentes**: ¿Cuál cita más fuentes externas?\n- **Nivel de confianza**: ¿Cuál muestra más certeza en su veredicto?\n- **Estilo de escritura**: ¿Cuál es más claro o profesional?""",
        "gemini_api_missing": "Clave de API de Gemini no configurada. Añade GEMINI_API_KEY en la configuración del Space.",
        "gemini_api_failed": "La solicitud a Gemini falló. Verifica la clave, el nombre del modelo o el cupo.",
    },
    "fr": {
        # App title and main UI
        "app_title": "FakeScope – Détecteur de Fausses Nouvelles",
        "analyze_tab": "🔍 Analyser",
        "chat_tab": "💬 Chat & Débat",
        "compare_tab": "⚖️ Comparer les Modèles",
        "deep_analysis_tab": "📊 Analyse Approfondie",
        "dashboard_tab": "📈 Tableau de Bord",
        # Language selector
        "language": "Langue",
        # Analyze section
        "analyze_subtitle": "Analyser un article ou une affirmation",
        "choose_ai_model": "Choisissez votre Modèle IA",
        "llm_provider": "Fournisseur LLM",
        "why_provider": "Pourquoi",
        "strengths": "Forces:",
        "cost": "Coût:",
        "article_url": "URL de l'article (optionnel)",
        "title_optional": "Titre (optionnel)",
        "article_text": "Texte de l'article ou affirmation",
        "fact_check_language": "Langue de Vérification",
        "auto_translate": "Traduire automatiquement en anglais",
        "fetch_from_url": "Extraire le texte de l'URL",
        "extracted_text": "Texte extrait",
        "run_analysis": "🚀 Lancer l'Analyse",
        "clear": "🗑️ Effacer",
        "provide_input_error": "⚠️ Veuillez fournir une URL, un titre ou un texte à analyser.",
        "analyzing": "🔄 Analyse avec",
        "extracted_success": "✅ Texte extrait de l'URL",
        "extract_failed": "⚠️ Impossible d'extraire le texte",
        # Results section
        "results": "📊 Résultats",
        "verdict": "Verdict:",
        "true": "VRAI",
        "fake": "FAUX",
        "credibility_score": "Score de Crédibilité",
        "fake_probability": "Probabilité Faux",
        "true_probability": "Probabilité Vrai",
        "translated_info": "✅ Le texte a été traduit de {lang} vers l'anglais pour l'analyse",
        # External fact checks
        "external_fact_checks": "🌐 Vérifications Externes",
        "google_fact_check_score": "Score Google Fact Check",
        "claim": "Affirmation",
        "rating": "Évaluation",
        "publisher": "Éditeur",
        "review_date": "Date de Révision",
        "no_fact_checks": "Aucune vérification externe trouvée pour cette affirmation.",
        # LLM Explanation
        "llm_explanation": "🤖 Explication IA",
        "provided_by": "Fourni par",
        # Chat section
        "start_conversation": "Démarrer une conversation sur l'analyse",
        "chat_subtitle": "Discuter et débattre de la crédibilité de votre affirmation analysée",
        "analyze_first": "👈 Veuillez d'abord effectuer une analyse dans l'onglet **Analyser**",
        "current_verdict": "Verdict Actuel",
        "your_message": "Votre message",
        "send": "Envoyer",
        "chat_history": "Historique du Chat",
        # Compare models
        "compare_subtitle": "Comparez les résultats des trois modèles IA côte à côte",
        "input_to_compare": "Entrez le texte à comparer entre les modèles",
        "compare_button": "⚖️ Comparer Tous les Modèles",
        "comparing": "Comparaison entre OpenAI, Gemini et Perplexity...",
        "model_comparison": "Résultats de Comparaison des Modèles",
        "model": "Modèle",
        "response": "Réponse",
        "response_time": "Temps de Réponse",
        "seconds": "secondes",
        # Deep analysis
        "deep_analysis_subtitle": "Obtenez une analyse complète avec sources et actualités connexes",
        "deep_analysis_button": "🔍 Lancer l'Analyse Approfondie",
        "deep_analyzing": "Exécution de l'analyse approfondie...",
        "sources_found": "Sources Trouvées",
        "related_articles": "Articles Connexes",
        "sentiment_analysis": "Analyse de Sentiment",
        "key_entities": "Entités Clés",
        # Dashboard
        "dashboard_subtitle": "Analyses récentes et statistiques",
        "recent_analyses": "Analyses Récentes",
        "no_data": "Aucune donnée disponible pour le moment. Effectuez d'abord quelques analyses!",
        "total_analyses": "Total des Analyses",
        "avg_credibility": "Crédibilité Moyenne",
        "most_used_provider": "Fournisseur le Plus Utilisé",
        # Provider details
        "provider_openai_description": "Idéal pour une analyse structurée et fiable. Excelle dans le suivi des instructions et la production d'explications bien formatées.",
        "provider_openai_strengths": "• Qualité constante\n• Réponses rapides (1-2s)\n• Excellent pour la vérification professionnelle",
        "provider_gemini_description": "Idéal pour des volumes élevés et des économies. Forfait gratuit avec 1 500 requêtes/jour. Rapide et compréhension naturelle du langage.",
        "provider_gemini_strengths": "• Forfait GRATUIT disponible\n• Réponses très rapides\n• Ton naturel et conversationnel\n• Multimodal",
        "provider_perplexity_description": "Idéal pour l'actualité et les nouvelles récentes. Inclut une recherche web en temps réel, fournissant un contexte à jour et des sources supplémentaires.",
        "provider_perplexity_strengths": "• Recherche web en temps réel\n• Informations les plus récentes\n• Cite automatiquement les sources\n• Excellent pour les dernières nouvelles",
        "view_source": "Voir la Source",
        # Chat quick prompts
        "quick_prompts_title": "💡 Suggestions Rapides",
        "quick_prompt_why": "Pourquoi est-ce faux/vrai ?",
        "quick_prompt_evidence": "Montre-moi des preuves",
        "quick_prompt_disagree": "Je ne suis pas d'accord",
        "quick_prompt_msg_why": "Explique en détail pourquoi tu penses que cette affirmation est fausse ou vraie.",
        "quick_prompt_msg_evidence": "Quelles preuves spécifiques soutiennent ou contredisent cette affirmation ?",
        "quick_prompt_msg_disagree": "Je ne suis pas d'accord avec ton évaluation. Peux-tu considérer des perspectives alternatives ?",
        # Deep analysis
        "source_analysis": "📰 Analyse des Sources",
        "found_fact_checks_count": "{count} vérifications externes trouvées",
        "rating_breakdown": "Répartition des Évaluations",
        "detailed_sources": "Sources Détaillées",
        "source_item": "Source {idx} : {publisher}",
        "rating_label": "Évaluation",
        "claim_label": "Affirmation",
        "url_label": "URL",
        "review_date_label": "Date de Révision",
        "no_external_sources": "Aucune source externe trouvée. Cette affirmation peut être trop récente ou trop spécifique.",
        "model_confidence": "🎯 Confiance du Modèle",
        "text_statistics": "📈 Statistiques du Texte",
        "characters": "Caractères",
        "words": "Mots",
        "sentences": "Phrases",
        "avg_word_length": "Longueur Moyenne des Mots",
        "key_topics": "🏷️ Sujets Clés",
        "no_keywords_identified": "Aucun mot-clé significatif identifié",
        "provider_changed_run_again": "Le fournisseur IA a changé. Cliquez sur 'Lancer l'Analyse' pour une nouvelle explication.",
        "key_differences_title": "🎯 Différences Clés",
        "key_differences_body": """Comparez comment chaque modèle aborde l'analyse :\n- **Profondeur de l'explication** : Lequel apporte le plus de détails ?\n- **Utilisation des sources** : Lequel cite le plus de sources externes ?\n- **Niveau de confiance** : Lequel est le plus sûr de son verdict ?\n- **Style d'écriture** : Lequel est le plus clair ou professionnel ?""",
        "gemini_api_missing": "Clé API Gemini non définie. Ajoutez GEMINI_API_KEY dans les paramètres du Space.",
        "gemini_api_failed": "Échec de la requête Gemini. Vérifiez la clé, le nom du modèle ou le quota.",
    },
    "de": {
        # App title and main UI
        "app_title": "FakeScope – Fake-News-Detektor",
        "analyze_tab": "🔍 Analysieren",
        "chat_tab": "💬 Chat & Debatte",
        "compare_tab": "⚖️ Modelle Vergleichen",
        "deep_analysis_tab": "📊 Tiefenanalyse",
        "dashboard_tab": "📈 Dashboard",
        # Language selector
        "language": "Sprache",
        # Analyze section
        "analyze_subtitle": "Einen Artikel oder eine Behauptung analysieren",
        "choose_ai_model": "Wählen Sie Ihr KI-Modell",
        "llm_provider": "LLM-Anbieter",
        "why_provider": "Warum",
        "strengths": "Stärken:",
        "cost": "Kosten:",
        "article_url": "Artikel-URL (optional)",
        "title_optional": "Titel (optional)",
        "article_text": "Artikeltext oder Behauptung",
        "fact_check_language": "Faktencheck-Sprache",
        "auto_translate": "Automatisch ins Englische übersetzen",
        "fetch_from_url": "Text von URL abrufen",
        "extracted_text": "Extrahierter Text",
        "run_analysis": "🚀 Analyse Starten",
        "clear": "🗑️ Löschen",
        "provide_input_error": "⚠️ Bitte geben Sie eine URL, einen Titel oder Text zur Analyse an.",
        "analyzing": "🔄 Analysiere mit",
        "extracted_success": "✅ Text von URL extrahiert",
        "extract_failed": "⚠️ Text konnte nicht extrahiert werden",
        # Results section
        "results": "📊 Ergebnisse",
        "verdict": "Urteil:",
        "true": "WAHR",
        "fake": "FALSCH",
        "credibility_score": "Glaubwürdigkeitswert",
        "fake_probability": "Wahrscheinlichkeit Falsch",
        "true_probability": "Wahrscheinlichkeit Wahr",
        "translated_info": "✅ Text wurde von {lang} ins Englische für die Analyse übersetzt",
        # External fact checks
        "external_fact_checks": "🌐 Externe Faktenchecks",
        "google_fact_check_score": "Google Fact Check Punktzahl",
        "claim": "Behauptung",
        "rating": "Bewertung",
        "publisher": "Herausgeber",
        "review_date": "Überprüfungsdatum",
        "no_fact_checks": "Keine externen Faktenchecks für diese Behauptung gefunden.",
        # LLM Explanation
        "llm_explanation": "🤖 KI-Erklärung",
        "provided_by": "Bereitgestellt von",
        # Chat section
        "start_conversation": "Starten Sie eine Unterhaltung über die Analyse",
        "chat_subtitle": "Diskutieren und debattieren Sie die Glaubwürdigkeit Ihrer analysierten Behauptung",
        "analyze_first": "👈 Bitte führen Sie zuerst eine Analyse im Tab **Analysieren** durch",
        "current_verdict": "Aktuelles Urteil",
        "your_message": "Ihre Nachricht",
        "send": "Senden",
        "chat_history": "Chat-Verlauf",
        # Compare models
        "compare_subtitle": "Vergleichen Sie Ergebnisse aller drei KI-Modelle nebeneinander",
        "input_to_compare": "Geben Sie Text ein, um zwischen Modellen zu vergleichen",
        "compare_button": "⚖️ Alle Modelle Vergleichen",
        "comparing": "Vergleiche zwischen OpenAI, Gemini und Perplexity...",
        "model_comparison": "Modellvergleich Ergebnisse",
        "model": "Modell",
        "response": "Antwort",
        "response_time": "Antwortzeit",
        "seconds": "Sekunden",
        # Deep analysis
        "deep_analysis_subtitle": "Erhalten Sie umfassende Analyse mit Quellen und verwandten Nachrichten",
        "deep_analysis_button": "🔍 Tiefenanalyse Starten",
        "deep_analyzing": "Führe Tiefenanalyse durch...",
        "sources_found": "Quellen Gefunden",
        "related_articles": "Verwandte Artikel",
        "sentiment_analysis": "Sentiment-Analyse",
        "key_entities": "Schlüsselentitäten",
        # Dashboard
        "dashboard_subtitle": "Aktuelle Analysen und Statistiken",
        "recent_analyses": "Aktuelle Analysen",
        "no_data": "Noch keine Daten verfügbar. Führen Sie zuerst einige Analysen durch!",
        "total_analyses": "Gesamtanalysen",
        "avg_credibility": "Durchschnittliche Glaubwürdigkeit",
        "most_used_provider": "Am Meisten Verwendeter Anbieter",
        # Provider details
        "provider_openai_description": "Am besten für strukturierte, zuverlässige Analysen. Hervorragend beim Befolgen von Anweisungen und beim Erstellen gut formatierter Erklärungen.",
        "provider_openai_strengths": "• Konstante Qualität\n• Schnelle Antworten (1-2s)\n• Hervorragend für professionelle Verifikation",
        "provider_gemini_description": "Am besten für hohes Volumen und Kosteneinsparungen. Kostenloses Kontingent mit 1.500 Anfragen/Tag. Schnell und mit natürlichem Sprachverständnis.",
        "provider_gemini_strengths": "• KOSTENLOSE Stufe verfügbar\n• Sehr schnelle Antworten\n• Natürlicher, konversationeller Ton\n• Multimodal fähig",
        "provider_perplexity_description": "Am besten für aktuelle Ereignisse und neueste Nachrichten. Beinhaltet Echtzeit-Websuche und liefert aktuelle Kontexte sowie zusätzliche Quellen.",
        "provider_perplexity_strengths": "• Echtzeit-Websuche\n• Neueste Informationen\n• Zitiert Quellen automatisch\n• Großartig für Eilmeldungen",
        "view_source": "Quelle anzeigen",
        # Chat quick prompts
        "quick_prompts_title": "💡 Schnelle Vorschläge",
        "quick_prompt_why": "Warum ist das falsch/wahr?",
        "quick_prompt_evidence": "Zeig mir Beweise",
        "quick_prompt_disagree": "Ich stimme nicht zu",
        "quick_prompt_msg_why": "Erläutere ausführlich, warum diese Behauptung deiner Meinung nach falsch oder wahr ist.",
        "quick_prompt_msg_evidence": "Welche konkreten Beweise stützen oder widerlegen diese Behauptung?",
        "quick_prompt_msg_disagree": "Ich stimme deiner Bewertung nicht zu. Kannst du alternative Perspektiven berücksichtigen?",
        # Deep analysis
        "source_analysis": "📰 Quellenanalyse",
        "found_fact_checks_count": "{count} externe Faktenchecks gefunden",
        "rating_breakdown": "Bewertungsübersicht",
        "detailed_sources": "Detaillierte Quellen",
        "source_item": "Quelle {idx}: {publisher}",
        "rating_label": "Bewertung",
        "claim_label": "Aussage",
        "url_label": "URL",
        "review_date_label": "Bewertungsdatum",
        "no_external_sources": "Keine externen Quellen gefunden. Diese Aussage ist möglicherweise zu neu oder zu spezifisch.",
        "model_confidence": "🎯 Modellvertrauen",
        "text_statistics": "📈 Textstatistiken",
        "characters": "Zeichen",
        "words": "Wörter",
        "sentences": "Sätze",
        "avg_word_length": "Durchschn. Wortlänge",
        "key_topics": "🏷️ Schlüsselthemen",
        "no_keywords_identified": "Keine bedeutenden Schlüsselwörter identifiziert",
        "provider_changed_run_again": "LLM-Anbieter geändert. Klicken Sie auf 'Analyse Starten' für eine neue Erklärung.",
        "key_differences_title": "🎯 Wichtige Unterschiede",
        "key_differences_body": """Vergleichen Sie, wie jedes Modell die Analyse angeht:\n- **Tiefe der Erklärung**: Welches bietet mehr Detail?\n- **Quellennutzung**: Welches zitiert mehr externe Quellen?\n- **Vertrauensniveau**: Welches ist sicherer im Urteil?\n- **Schreibstil**: Welches ist klarer oder professioneller?""",
        "gemini_api_missing": "Gemini API-Schlüssel nicht gesetzt. Fügen Sie GEMINI_API_KEY in den Space-Einstellungen hinzu.",
        "gemini_api_failed": "Gemini-Anfrage fehlgeschlagen. Schlüssel, Modellname oder Kontingent prüfen.",
    },
    "ru": {
        # App title and main UI
        "app_title": "FakeScope – Детектор Фейковых Новостей",
        "analyze_tab": "🔍 Анализ",
        "chat_tab": "💬 Чат и Дебаты",
        "compare_tab": "⚖️ Сравнить Модели",
        "deep_analysis_tab": "📊 Глубокий Анализ",
        "dashboard_tab": "📈 Панель",
        # Language selector
        "language": "Язык",
        # Analyze section
        "analyze_subtitle": "Анализировать статью или утверждение",
        "choose_ai_model": "Выберите вашу AI Модель",
        "llm_provider": "Провайдер LLM",
        "why_provider": "Почему",
        "strengths": "Преимущества:",
        "cost": "Стоимость:",
        "article_url": "URL статьи (необязательно)",
        "title_optional": "Заголовок (необязательно)",
        "article_text": "Текст статьи или утверждение",
        "fact_check_language": "Язык Проверки Фактов",
        "auto_translate": "Автоматически переводить на английский",
        "fetch_from_url": "Извлечь текст из URL",
        "extracted_text": "Извлечённый текст",
        "run_analysis": "🚀 Запустить Анализ",
        "clear": "🗑️ Очистить",
        "provide_input_error": "⚠️ Пожалуйста, предоставьте URL, заголовок или текст для анализа.",
        "analyzing": "🔄 Анализ с помощью",
        "extracted_success": "✅ Текст извлечён из URL",
        "extract_failed": "⚠️ Не удалось извлечь текст",
        # Results section
        "results": "📊 Результаты",
        "verdict": "Вердикт:",
        "true": "ПРАВДА",
        "fake": "ЛОЖЬ",
        "credibility_score": "Оценка Достоверности",
        "fake_probability": "Вероятность Лжи",
        "true_probability": "Вероятность Правды",
        "translated_info": "✅ Текст был переведён с {lang} на английский для анализа",
        # External fact checks
        "external_fact_checks": "🌐 Внешние Проверки Фактов",
        "google_fact_check_score": "Оценка Google Fact Check",
        "claim": "Утверждение",
        "rating": "Рейтинг",
        "publisher": "Издатель",
        "review_date": "Дата Проверки",
        "no_fact_checks": "Внешние проверки фактов для этого утверждения не найдены.",
        # LLM Explanation
        "llm_explanation": "🤖 Объяснение AI",
        "provided_by": "Предоставлено",
        # Chat section
        "start_conversation": "Начать разговор об анализе",
        "chat_subtitle": "Обсудить и оспорить достоверность вашего проанализированного утверждения",
        "analyze_first": "👈 Пожалуйста, сначала выполните анализ на вкладке **Анализ**",
        "current_verdict": "Текущий Вердикт",
        "your_message": "Ваше сообщение",
        "send": "Отправить",
        "chat_history": "История Чата",
        # Compare models
        "compare_subtitle": "Сравните результаты всех трёх AI моделей",
        "input_to_compare": "Введите текст для сравнения между моделями",
        "compare_button": "⚖️ Сравнить Все Модели",
        "comparing": "Сравнение между OpenAI, Gemini и Perplexity...",
        "model_comparison": "Результаты Сравнения Моделей",
        "model": "Модель",
        "response": "Ответ",
        "response_time": "Время Ответа",
        "seconds": "секунд",
        # Deep analysis
        "deep_analysis_subtitle": "Получите полный анализ с источниками и связанными новостями",
        "deep_analysis_button": "🔍 Запустить Глубокий Анализ",
        "deep_analyzing": "Выполнение глубокого анализа...",
        "sources_found": "Найдено Источников",
        "related_articles": "Связанные Статьи",
        "sentiment_analysis": "Анализ Тональности",
        "key_entities": "Ключевые Сущности",
        # Dashboard
        "dashboard_subtitle": "Последние анализы и статистика",
        "recent_analyses": "Последние Анализы",
        "no_data": "Данные пока недоступны. Сначала выполните несколько анализов!",
        "total_analyses": "Всего Анализов",
        "avg_credibility": "Средняя Достоверность",
        "most_used_provider": "Наиболее Используемый Провайдер",
        # Provider details
        "provider_openai_description": "Лучше всего подходит для структурированного и надежного анализа. Отлично следует инструкциям и создает хорошо оформленные объяснения.",
        "provider_openai_strengths": "• Стабильное качество\n• Быстрые ответы (1-2с)\n• Отлично для профессиональной проверки",
        "provider_gemini_description": "Подходит для большого объема и экономии. Бесплатный тариф с 1 500 запросами/день. Быстрый и хорошо понимает естественный язык.",
        "provider_gemini_strengths": "• Доступен БЕСПЛАТНЫЙ тариф\n• Очень быстрые ответы\n• Естественный, разговорный тон\n• Мультимодальные возможности",
        "provider_perplexity_description": "Лучший выбор для текущих событий и последних новостей. Включает поиск в интернете в реальном времени, предоставляя актуальный контекст и дополнительные источники.",
        "provider_perplexity_strengths": "• Поиск в реальном времени\n• Самая свежая информация\n• Автоматически цитирует источники\n• Отлично для срочных новостей",
        "view_source": "Открыть источник",
        # Chat quick prompts
        "quick_prompts_title": "💡 Быстрые Подсказки",
        "quick_prompt_why": "Почему это ложь/правда?",
        "quick_prompt_evidence": "Покажи доказательства",
        "quick_prompt_disagree": "Я не согласен",
        "quick_prompt_msg_why": "Подробно объясните, почему вы считаете это утверждение ложным или истинным.",
        "quick_prompt_msg_evidence": "Какие конкретные доказательства подтверждают или опровергают это утверждение?",
        "quick_prompt_msg_disagree": "Я не согласен с вашей оценкой. Можете рассмотреть альтернативные точки зрения?",
        # Deep analysis
        "source_analysis": "📰 Анализ Источников",
        "found_fact_checks_count": "Найдено внешних проверок фактов: {count}",
        "rating_breakdown": "Распределение Оценок",
        "detailed_sources": "Подробные Источники",
        "source_item": "Источник {idx}: {publisher}",
        "rating_label": "Рейтинг",
        "claim_label": "Утверждение",
        "url_label": "URL",
        "review_date_label": "Дата Проверки",
        "no_external_sources": "Внешние источники не найдены. Возможно, утверждение слишком новое или слишком специфичное.",
        "model_confidence": "🎯 Уверенность Модели",
        "text_statistics": "📈 Статистика Текста",
        "characters": "Символы",
        "words": "Слова",
        "sentences": "Предложения",
        "avg_word_length": "Средняя Длина Слова",
        "key_topics": "🏷️ Ключевые Темы",
        "no_keywords_identified": "Значимые ключевые слова не обнаружены",
        "provider_changed_run_again": "Провайдер LLM изменён. Нажмите 'Запустить Анализ' для нового объяснения.",
        "key_differences_title": "🎯 Ключевые Отличия",
        "key_differences_body": """Сравните подход моделей к анализу:\n- **Глубина объяснения**: Какая даёт больше деталей?\n- **Использование источников**: Какая цитирует больше внешних источников?\n- **Уровень уверенности**: Какая более уверена в вердикте?\n- **Стиль письма**: Какая яснее или профессиональнее?""",
        "gemini_api_missing": "API-ключ Gemini не задан. Добавьте GEMINI_API_KEY в настройках Space.",
        "gemini_api_failed": "Запрос к Gemini не удался. Проверьте ключ, имя модели или квоту.",
    },
    "pt": {
        # App title and main UI
        "app_title": "FakeScope – Detector de Notícias Falsas",
        "analyze_tab": "🔍 Analisar",
        "chat_tab": "💬 Chat e Debate",
        "compare_tab": "⚖️ Comparar Modelos",
        "deep_analysis_tab": "📊 Análise Profunda",
        "dashboard_tab": "📈 Painel",
        # Language selector
        "language": "Idioma",
        # Analyze section
        "analyze_subtitle": "Analisar um artigo ou afirmação",
        "choose_ai_model": "Escolha seu Modelo de IA",
        "llm_provider": "Provedor LLM",
        "why_provider": "Por que",
        "strengths": "Pontos Fortes:",
        "cost": "Custo:",
        "article_url": "URL do artigo (opcional)",
        "title_optional": "Título (opcional)",
        "article_text": "Texto do artigo ou afirmação",
        "fact_check_language": "Idioma de Verificação",
        "auto_translate": "Traduzir automaticamente para inglês",
        "fetch_from_url": "Obter texto da URL",
        "extracted_text": "Texto extraído",
        "run_analysis": "🚀 Executar Análise",
        "clear": "🗑️ Limpar",
        "provide_input_error": "⚠️ Por favor, forneça uma URL, título ou texto para analisar.",
        "analyzing": "🔄 Analisando com",
        "extracted_success": "✅ Texto extraído da URL",
        "extract_failed": "⚠️ Não foi possível extrair o texto",
        # Results section
        "results": "📊 Resultados",
        "verdict": "Veredicto:",
        "true": "VERDADEIRO",
        "fake": "FALSO",
        "credibility_score": "Pontuação de Credibilidade",
        "fake_probability": "Probabilidade Falso",
        "true_probability": "Probabilidade Verdadeiro",
        "translated_info": "✅ O texto foi traduzido de {lang} para inglês para análise",
        # External fact checks
        "external_fact_checks": "🌐 Verificações Externas",
        "google_fact_check_score": "Pontuação Google Fact Check",
        "claim": "Afirmação",
        "rating": "Classificação",
        "publisher": "Editor",
        "review_date": "Data de Revisão",
        "no_fact_checks": "Nenhuma verificação externa encontrada para esta afirmação.",
        # LLM Explanation
        "llm_explanation": "🤖 Explicação de IA",
        "provided_by": "Fornecido por",
        # Chat section
        "start_conversation": "Iniciar uma conversa sobre a análise",
        "chat_subtitle": "Discutir e debater a credibilidade da sua afirmação analisada",
        "analyze_first": "👈 Por favor, execute uma análise primeiro na aba **Analisar**",
        "current_verdict": "Veredicto Atual",
        "your_message": "Sua mensagem",
        "send": "Enviar",
        "chat_history": "Histórico de Chat",
        # Compare models
        "compare_subtitle": "Compare resultados dos três modelos de IA lado a lado",
        "input_to_compare": "Digite texto para comparar entre modelos",
        "compare_button": "⚖️ Comparar Todos os Modelos",
        "comparing": "Comparando entre OpenAI, Gemini e Perplexity...",
        "model_comparison": "Resultados de Comparação de Modelos",
        "model": "Modelo",
        "response": "Resposta",
        "response_time": "Tempo de Resposta",
        "seconds": "segundos",
        # Deep analysis
        "deep_analysis_subtitle": "Obtenha análise abrangente com fontes e notícias relacionadas",
        "deep_analysis_button": "🔍 Executar Análise Profunda",
        "deep_analyzing": "Executando análise profunda...",
        "sources_found": "Fontes Encontradas",
        "related_articles": "Artigos Relacionados",
        "sentiment_analysis": "Análise de Sentimento",
        "key_entities": "Entidades Principais",
        # Dashboard
        "dashboard_subtitle": "Análises recentes e estatísticas",
        "recent_analyses": "Análises Recentes",
        "no_data": "Nenhum dado disponível ainda. Execute algumas análises primeiro!",
        "total_analyses": "Total de Análises",
        "avg_credibility": "Credibilidade Média",
        "most_used_provider": "Provedor Mais Usado",
        # Provider details
        "provider_openai_description": "Melhor para análises estruturadas e confiáveis. Excelente em seguir instruções e gerar explicações bem formatadas.",
        "provider_openai_strengths": "• Qualidade consistente\n• Respostas rápidas (1-2s)\n• Excelente para verificação profissional",
        "provider_gemini_description": "Melhor para alto volume e economia. Camada gratuita com 1.500 solicitações/dia. Rápido e com compreensão natural da linguagem.",
        "provider_gemini_strengths": "• Camada GRÁTIS disponível\n• Respostas muito rápidas\n• Tom natural e conversacional\n• Capaz de multimodal",
        "provider_perplexity_description": "Melhor para eventos atuais e notícias recentes. Inclui busca na web em tempo real, fornecendo contexto atualizado e fontes adicionais.",
        "provider_perplexity_strengths": "• Busca na web em tempo real\n• Informações mais recentes\n• Cita fontes automaticamente\n• Ótimo para notícias de última hora",
        "view_source": "Ver Fonte",
        # Chat quick prompts
        "quick_prompts_title": "💡 Sugestões Rápidas",
        "quick_prompt_why": "Por que é falso/verdadeiro?",
        "quick_prompt_evidence": "Mostre-me evidências",
        "quick_prompt_disagree": "Eu discordo",
        "quick_prompt_msg_why": "Explique em detalhe por que você acha que esta afirmação é falsa ou verdadeira.",
        "quick_prompt_msg_evidence": "Que evidências específicas apoiam ou contradizem esta afirmação?",
        "quick_prompt_msg_disagree": "Eu discordo da sua avaliação. Você pode considerar perspectivas alternativas?",
        # Deep analysis
        "source_analysis": "📰 Análise de Fontes",
        "found_fact_checks_count": "{count} verificações externas encontradas",
        "rating_breakdown": "Distribuição de Classificações",
        "detailed_sources": "Fontes Detalhadas",
        "source_item": "Fonte {idx}: {publisher}",
        "rating_label": "Classificação",
        "claim_label": "Afirmação",
        "url_label": "URL",
        "review_date_label": "Data de Revisão",
        "no_external_sources": "Nenhuma fonte externa encontrada. Esta afirmação pode ser muito recente ou muito específica.",
        "model_confidence": "🎯 Confiança do Modelo",
        "text_statistics": "📈 Estatísticas do Texto",
        "characters": "Caracteres",
        "words": "Palavras",
        "sentences": "Frases",
        "avg_word_length": "Tamanho Médio da Palavra",
        "key_topics": "🏷️ Tópicos Chave",
        "no_keywords_identified": "Nenhuma palavra-chave significativa identificada",
        "provider_changed_run_again": "Provedor LLM alterado. Clique em 'Executar Análise' para uma nova explicação.",
        "key_differences_title": "🎯 Diferenças Principais",
        "key_differences_body": """Compare como cada modelo aborda a análise:\n- **Profundidade da explicação**: Qual fornece mais detalhes?\n- **Uso de fontes**: Qual cita mais fontes externas?\n- **Nível de confiança**: Qual demonstra mais certeza no veredicto?\n- **Estilo de escrita**: Qual é mais claro ou profissional?""",
        "gemini_api_missing": "Chave da API Gemini não definida. Adicione GEMINI_API_KEY nas configurações do Space.",
        "gemini_api_failed": "Falha na requisição Gemini. Verifique a chave, nome do modelo ou quota.",
    },
}

# Country code to language mapping for IP-based detection
COUNTRY_TO_LANGUAGE = {
    "ES": "es",  # Spain
    "MX": "es",  # Mexico
    "AR": "es",  # Argentina
    "CO": "es",  # Colombia
    "PE": "es",  # Peru
    "VE": "es",  # Venezuela
    "CL": "es",  # Chile
    "EC": "es",  # Ecuador
    "GT": "es",  # Guatemala
    "CU": "es",  # Cuba
    "BO": "es",  # Bolivia
    "DO": "es",  # Dominican Republic
    "HN": "es",  # Honduras
    "PY": "es",  # Paraguay
    "SV": "es",  # El Salvador
    "NI": "es",  # Nicaragua
    "CR": "es",  # Costa Rica
    "PA": "es",  # Panama
    "UY": "es",  # Uruguay
    "FR": "fr",  # France
    "BE": "fr",  # Belgium
    "CH": "fr",  # Switzerland
    "LU": "fr",  # Luxembourg
    "MC": "fr",  # Monaco
    "CA": "fr",  # Canada (partial)
    "DE": "de",  # Germany
    "AT": "de",  # Austria
    "LI": "de",  # Liechtenstein
    "RU": "ru",  # Russia
    "BY": "ru",  # Belarus
    "KZ": "ru",  # Kazakhstan
    "KG": "ru",  # Kyrgyzstan
    "PT": "pt",  # Portugal
    "BR": "pt",  # Brazil
    "AO": "pt",  # Angola
    "MZ": "pt",  # Mozambique
}

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "ru", "pt"]


def get_country_from_ip(ip: Optional[str] = None) -> Optional[str]:
    """
    Detect country from IP address using ip-api.com (free, no key required).
    Returns ISO country code (e.g., 'US', 'ES', 'FR') or None if detection fails.
    """
    if not ip or ip in ["127.0.0.1", "localhost", "::1"]:
        return None

    try:
        # Use ip-api.com free tier (45 requests/minute limit)
        response = requests.get(f"http://ip-api.com/json/{ip}", timeout=2)
        if response.ok:
            data = response.json()
            if data.get("status") == "success":
                return data.get("countryCode")
    except Exception:
        pass

    return None


def detect_language_from_ip(ip: Optional[str] = None) -> str:
    """
    Detect preferred language based on visitor's IP address.
    Returns language code ('en', 'es', 'fr', 'de', 'ru', 'pt') with 'en' as default.
    """
    country = get_country_from_ip(ip)
    if country and country in COUNTRY_TO_LANGUAGE:
        lang = COUNTRY_TO_LANGUAGE[country]
        if lang in SUPPORTED_LANGUAGES:
            return lang

    return "en"  # Default to English


def get_translation(key: str, language: str = "en", **kwargs) -> str:
    """
    Get translated text for a given key and language.

    Args:
        key: Translation key (e.g., 'app_title', 'analyze_tab')
        language: Language code ('en', 'es', 'fr', 'de', 'ru', 'pt')
        **kwargs: Variables to format into the translation (e.g., lang='ES')

    Returns:
        Translated string, falling back to English if translation not found
    """
    if language not in TRANSLATIONS:
        language = "en"

    text = TRANSLATIONS[language].get(key, TRANSLATIONS["en"].get(key, key))

    # Apply formatting if kwargs provided
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass

    return text


def get_language_name(code: str) -> str:
    """Get the full language name for a language code."""
    names = {
        "en": "English",
        "es": "Español",
        "fr": "Français",
        "de": "Deutsch",
        "ru": "Русский",
        "pt": "Português",
    }
    return names.get(code, code)
