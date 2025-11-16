import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from deep_translator import GoogleTranslator
except Exception as e:
    GoogleTranslator = None  # Library not available
    logger.warning(f"deep-translator not available: {e}")


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English if source_lang != 'en'.
    Falls back to original text on any failure or if translation disabled.
    Set env FAKESCOPE_DISABLE_TRANSLATION=1 to bypass.
    """
    if not text or source_lang.lower() == "en":
        logger.debug(f"Skipping translation: text={bool(text)}, lang={source_lang}")
        return text
    if os.getenv("FAKESCOPE_DISABLE_TRANSLATION") == "1":
        logger.info("Translation disabled by env var")
        return text
    if GoogleTranslator is None:
        logger.warning("GoogleTranslator not available")
        return text
    try:
        logger.info(f"Translating from {source_lang} to en, text length: {len(text)}")
        translator = GoogleTranslator(source=source_lang, target="en")
        translated = translator.translate(text)
        if isinstance(translated, str) and translated.strip():
            logger.info(f"Translation successful, output length: {len(translated)}")
            return translated
        else:
            logger.warning(f"Translation returned invalid result: {type(translated)}")
    except Exception as e:
        logger.error(f"Translation failed: {e}")
    return text
