import pytest

from swahili_nlp.preprocessing import SwahiliTextPreprocessor


@pytest.fixture
def preprocessor() -> SwahiliTextPreprocessor:
    return SwahiliTextPreprocessor()


def test_normalize_lowercases_and_strips(preprocessor: SwahiliTextPreprocessor) -> None:
    assert preprocessor.normalize("  Habari YA Leo  ") == "habari ya leo"


def test_normalize_rejects_non_string(preprocessor: SwahiliTextPreprocessor) -> None:
    with pytest.raises(TypeError):
        preprocessor.normalize(12345)  # type: ignore[arg-type]


def test_tokenize_removes_punctuation_and_digits(preprocessor: SwahiliTextPreprocessor) -> None:
    tokens = preprocessor.tokenize("Bei ya mafuta imepanda kwa 20% mwaka 2026!")
    assert "20" not in tokens
    assert "2026" not in tokens
    assert "%" not in "".join(tokens)


def test_tokenize_removes_stopwords_by_default(preprocessor: SwahiliTextPreprocessor) -> None:
    tokens = preprocessor.tokenize("Rais na waziri wa mambo ya ndani")
    assert "na" not in tokens
    assert "wa" not in tokens
    assert "ya" not in tokens
    assert "rais" in tokens
    assert "waziri" in tokens


def test_tokenize_can_keep_stopwords() -> None:
    pre = SwahiliTextPreprocessor(remove_stopwords=False)
    tokens = pre.tokenize("Rais na waziri")
    assert "na" in tokens


def test_min_token_length_filters_short_tokens() -> None:
    pre = SwahiliTextPreprocessor(remove_stopwords=False, min_token_length=3)
    tokens = pre.tokenize("a bb ccc dddd")
    assert tokens == ["ccc", "dddd"]


def test_preprocess_returns_joined_string(preprocessor: SwahiliTextPreprocessor) -> None:
    result = preprocessor.preprocess("Rais ametangaza bajeti mpya ya Serikali!")
    assert isinstance(result, str)
    assert "serikali" in result
    assert "ya" not in result.split()


def test_preprocess_batch(preprocessor: SwahiliTextPreprocessor) -> None:
    results = preprocessor.preprocess_batch(["Rais ametangaza bajeti", "Timu imeshinda mechi"])
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_empty_string_returns_empty_tokens(preprocessor: SwahiliTextPreprocessor) -> None:
    assert preprocessor.tokenize("") == []
    assert preprocessor.tokenize("   ") == []
