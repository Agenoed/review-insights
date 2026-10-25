from lingua import Language, LanguageDetectorBuilder

UA_LETTERS = set("іїєґІЇЄҐ")
RU_LETTERS = set("ыэёъЫЭЁЪ")
LINGUA_CODES = {
    Language.UKRAINIAN: "uk",
    Language.RUSSIAN: "ru",
    Language.ENGLISH: "en",
    Language.POLISH: "pl",
}

detector = LanguageDetectorBuilder.from_languages(*LINGUA_CODES).build()


def lang_by_letters(text):
    has_ua = any(ch in UA_LETTERS for ch in text)
    has_ru = any(ch in RU_LETTERS for ch in text)
    if has_ua and has_ru:
        return "mixed"
    if has_ua:
        return "uk"
    if has_ru:
        return "ru"
    return None


def detect_lang(text):
    if not any(ch.isalpha() for ch in text):
        return "no_letters"
    lang = lang_by_letters(text)
    if lang:
        return lang
    result = detector.detect_language_of(text)
    if result is None:
        return "unknown"
    return LINGUA_CODES[result]