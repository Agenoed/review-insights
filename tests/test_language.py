from src.preprocessing.language import detect_lang, lang_by_letters


def test_letters_ukrainian():
    assert lang_by_letters("Дуже зручно, дякую за підтримку") == "uk"


def test_letters_russian():
    assert lang_by_letters("Отличное приложение, всё работает") == "ru"


def test_both_alphabets_go_to_detector():
    assert lang_by_letters("Все працює, всё отлично") is None


def test_no_letters():
    assert detect_lang("👍👍 !!!") == "no_letters"


def test_detector_ukrainian_without_special_letters():
    assert detect_lang("Дуже зручний застосунок, рекомендую") == "uk"


def test_detector_russian_without_special_letters():
    assert detect_lang("Отличное приложение, рекомендую") == "ru"


def test_detector_english():
    assert detect_lang("Great app, works fine") == "en"


def test_mixed_text_gets_one_language():
    assert detect_lang("Все працює, всё отлично") in {"uk", "ru"}