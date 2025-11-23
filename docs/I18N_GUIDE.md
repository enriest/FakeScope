# Internationalization (i18n) Guide

## Overview
FakeScope now supports 6 languages with automatic IP-based detection and manual language switching:
- **English (en)** - Default
- **Español (es)** - Spanish
- **Français (fr)** - French
- **Deutsch (de)** - German
- **Русский (ru)** - Russian
- **Português (pt)** - Portuguese

## Features

### 1. Automatic Language Detection
When a user visits the app for the first time, the system attempts to detect their country from their IP address and automatically sets the appropriate language:

- **Spanish-speaking countries**: Spain, Mexico, Argentina, Colombia, Peru, Venezuela, Chile, Ecuador, Guatemala, Cuba, Bolivia, Dominican Republic, Honduras, Paraguay, El Salvador, Nicaragua, Costa Rica, Panama, Uruguay
- **French-speaking countries**: France, Belgium, Switzerland, Luxembourg, Monaco, Canada (partial)
- **German-speaking countries**: Germany, Austria, Liechtenstein
- **Russian-speaking countries**: Russia, Belarus, Kazakhstan, Kyrgyzstan
- **Portuguese-speaking countries**: Portugal, Brazil, Angola, Mozambique

If detection fails or the user is from an unsupported country, the app defaults to English.

### 2. Manual Language Selector
Users can manually change the language at any time using the dropdown selector in the top-right corner of the app. The change takes effect immediately across all tabs.

### 3. Translated UI Elements
All major UI elements are translated, including:
- **App title and tab names**
- **Form labels and buttons**
- **Instructions and help text**
- **Results and metrics**
- **Error messages and notifications**
- **Chat interface**
- **Model comparison labels**
- **Deep analysis sections**
- **Dashboard headings**

## Technical Implementation

### Architecture
The internationalization system consists of two main components:

#### 1. `src/i18n.py`
Core module containing:
- **Translation dictionaries** (`TRANSLATIONS`): Complete translations for all UI strings in 6 languages
- **Country-to-language mapping** (`COUNTRY_TO_LANGUAGE`): Maps ISO country codes to language codes
- **IP detection** (`get_country_from_ip`): Uses ip-api.com free API to geolocate visitors
- **Language detection** (`detect_language_from_ip`): Determines preferred language from IP
- **Translation helper** (`get_translation`): Retrieves translated strings with formatting support

#### 2. `src/app.py` Updates
- Session state management for `ui_language`
- Language selector in app header
- Translation helper function `t()` for quick lookups
- All UI strings replaced with `t("key")` calls

### Usage in Code

```python
from src.i18n import get_translation, detect_language_from_ip

# Basic translation
current_lang = "es"
title = get_translation("app_title", current_lang)
# Returns: "FakeScope – Detector de Noticias Falsas"

# With formatting
msg = get_translation("translated_info", current_lang, lang="ES")
# Returns: "✅ El texto fue traducido de ES al inglés para el análisis"

# IP-based detection
detected_lang = detect_language_from_ip("8.8.8.8")  # Example IP
# Returns: "en" (default for US)
```

### Adding New Languages

To add a new language:

1. Add the language code to `SUPPORTED_LANGUAGES` in `src/i18n.py`
2. Add country mappings to `COUNTRY_TO_LANGUAGE`
3. Create a complete translation dictionary in `TRANSLATIONS`
4. Add the language name to `get_language_name()`

Example:
```python
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "ru", "pt", "it"]  # Add Italian

COUNTRY_TO_LANGUAGE = {
    # ... existing mappings ...
    "IT": "it",  # Italy
}

TRANSLATIONS = {
    # ... existing translations ...
    "it": {
        "app_title": "FakeScope – Rilevatore di Notizie False",
        "analyze_tab": "🔍 Analizza",
        # ... complete all keys ...
    }
}
```

### Translation Keys Reference

Key translation keys organized by section:

**Core UI:**
- `app_title`, `analyze_tab`, `chat_tab`, `compare_tab`, `deep_analysis_tab`, `dashboard_tab`
- `language` (for language selector)

**Analyze Tab:**
- `analyze_subtitle`, `choose_ai_model`, `llm_provider`, `article_url`, `title_optional`, `article_text`
- `fact_check_language`, `auto_translate`, `run_analysis`, `clear`

**Results:**
- `results`, `verdict`, `true`, `fake`, `credibility_score`, `fake_probability`, `true_probability`
- `external_fact_checks`, `google_fact_check_score`, `no_fact_checks`

**Chat:**
- `chat_subtitle`, `analyze_first`, `current_verdict`, `your_message`, `send`

**Compare & Deep Analysis:**
- `compare_subtitle`, `model`, `compare_button`, `comparing`
- `deep_analysis_subtitle`, `sources_found`, `related_articles`

**Dashboard:**
- `dashboard_subtitle`, `recent_analyses`, `no_data`, `total_analyses`, `avg_credibility`

## IP Detection Service

### Free Tier Limits
The app uses **ip-api.com** for IP geolocation (free tier):
- **45 requests per minute**
- **No API key required**
- **Commercial use allowed**

### Fallback Behavior
If IP detection fails (local development, rate limits, network errors), the app defaults to English. Users can always manually select their preferred language.

### Privacy Considerations
- IP addresses are **not stored** or logged
- Geolocation happens **client-side** on first load only
- The detected country is used solely for language preference
- Users can override the automatic selection at any time

## Testing Language Support

### Local Testing
To test different languages locally:

1. **Manual selection**: Use the language dropdown in the UI
2. **Force detection**: Temporarily modify `detect_language_from_ip()` to return a specific language code

```python
# In src/i18n.py, temporarily modify:
def detect_language_from_ip(ip: Optional[str] = None) -> str:
    return "es"  # Force Spanish for testing
```

3. **Session state reset**: Clear browser cache or use incognito mode to simulate first-time visitor

### Automated Testing
Create a test script:

```python
# test_i18n.py
from src.i18n import get_translation, SUPPORTED_LANGUAGES

def test_all_translations():
    required_keys = ["app_title", "analyze_tab", "run_analysis", "verdict", "true", "fake"]
    
    for lang in SUPPORTED_LANGUAGES:
        print(f"\nTesting {lang}:")
        for key in required_keys:
            translation = get_translation(key, lang)
            print(f"  {key}: {translation}")
            assert translation, f"Missing translation for {key} in {lang}"

if __name__ == "__main__":
    test_all_translations()
    print("\n✅ All translations present!")
```

## Deployment Considerations

### Streamlit Cloud
When deploying to Streamlit Cloud, IP detection works automatically via request headers:
- `X-Forwarded-For` header contains client IP
- `X-Real-Ip` header as fallback

### Docker/Custom Deployment
Ensure your reverse proxy passes through client IP:

**Nginx example:**
```nginx
location / {
    proxy_pass http://streamlit:8501;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Real-Ip $remote_addr;
}
```

**Caddy example:**
```
reverse_proxy streamlit:8501
```
(Caddy automatically forwards real IP)

### Rate Limiting
If your app receives >45 new visitors per minute, consider:
1. **Caching** detection results by IP (with TTL)
2. **Alternative services**: ipapi.co, ipgeolocation.io (paid tiers)
3. **Browser locale detection**: Use JavaScript to detect `navigator.language`

## Maintenance

### Updating Translations
When adding new UI strings:

1. Add the English version first to `TRANSLATIONS["en"]`
2. Use a clear, descriptive key (e.g., `new_feature_button`)
3. Update all other language dictionaries with translations
4. Use placeholders for dynamic content: `"Welcome {name}"`
5. Test with `get_translation("new_feature_button", "es")`

### Translation Quality
For production use, consider:
- **Professional translation services** for accuracy
- **Native speaker review** for cultural appropriateness
- **A/B testing** different phrasings
- **User feedback** mechanism for reporting translation issues

## Examples

### Spanish User Flow
1. User from Madrid (IP → ES) visits app
2. App detects country code "ES" → language "es"
3. UI displays: "FakeScope – Detector de Noticias Falsas"
4. Tabs show: "🔍 Analizar", "💬 Chat y Debate", etc.
5. User can switch to "English" if preferred

### Manual Override
1. User from US (defaults to English)
2. Selects "Deutsch" from language dropdown
3. All UI instantly switches to German
4. Analysis results in German: "WAHR" instead of "TRUE"
5. Preference persists in session

## Best Practices

1. **Always use translation keys**, never hardcode strings
2. **Keep keys semantic**: `submit_button` not `button_1`
3. **Group related keys**: `chat_title`, `chat_subtitle`, `chat_send_button`
4. **Provide context** in comments for translators
5. **Test RTL languages** (Arabic, Hebrew) if adding support
6. **Use Unicode** for non-Latin scripts (Cyrillic, Chinese, etc.)
7. **Avoid concatenation**: Use placeholders instead

## Troubleshooting

### Issue: Language not detecting correctly
**Solution:** Check IP detection service availability:
```bash
curl "http://ip-api.com/json/8.8.8.8"
```

### Issue: Missing translations show keys
**Solution:** Verify key exists in all language dictionaries:
```python
from src.i18n import TRANSLATIONS
key = "missing_key"
for lang in TRANSLATIONS:
    if key not in TRANSLATIONS[lang]:
        print(f"Missing: {key} in {lang}")
```

### Issue: Language selector not persisting
**Solution:** Ensure session state is initialized before any UI rendering:
```python
if "ui_language" not in st.session_state:
    st.session_state.ui_language = detect_language_from_ip()
```

## Future Enhancements

Potential improvements:
- **Browser locale fallback**: Use `Accept-Language` header when IP detection fails
- **User preference storage**: Remember language choice in cookies/localStorage
- **LLM response translation**: Translate AI explanations to match UI language
- **More languages**: Add Chinese, Japanese, Arabic, Hindi
- **Crowdsourced translations**: Community contribution platform
- **A/B testing framework**: Experiment with different phrasings

---

For questions or translation contributions, please open an issue on GitHub.
