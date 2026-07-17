import tempfile
from pathlib import Path

import pytest

from swahili_nlp.classifier import (
    NotFittedError,
    PredictionResult,
    SwahiliNewsClassifier,
)

TRAIN_TEXTS = [
    "Rais ametangaza sera mpya ya elimu",
    "Waziri Mkuu amefanya ziara ya kikazi",
    "Bunge limepitisha sheria mpya",
    "Timu ya taifa imeshinda mechi ya kirafiki",
    "Kocha mkuu ameteuliwa kuwa nahodha",
    "Klabu ya mpira imepanda daraja",
    "Benki kuu imeongeza thamani ya sarafu",
    "Soko la hisa limeripoti ukuaji wa uchumi",
    "Kampuni kubwa imetangaza faida kubwa",
]
TRAIN_LABELS = [
    "siasa", "siasa", "siasa",
    "michezo", "michezo", "michezo",
    "biashara", "biashara", "biashara",
]


@pytest.fixture
def fitted_classifier() -> SwahiliNewsClassifier:
    clf = SwahiliNewsClassifier(max_features=200)
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    return clf


def test_fit_raises_on_empty_input() -> None:
    clf = SwahiliNewsClassifier()
    with pytest.raises(ValueError):
        clf.fit([], [])


def test_fit_raises_on_mismatched_lengths() -> None:
    clf = SwahiliNewsClassifier()
    with pytest.raises(ValueError):
        clf.fit(["a", "b"], ["only-one-label"])


def test_predict_before_fit_raises() -> None:
    clf = SwahiliNewsClassifier()
    with pytest.raises(NotFittedError):
        clf.predict("Rais ametangaza sera mpya")


def test_predict_returns_known_label(fitted_classifier: SwahiliNewsClassifier) -> None:
    label = fitted_classifier.predict("Bunge limepitisha sheria mpya ya kodi")
    assert label in {"siasa", "michezo", "biashara"}


def test_predict_proba_returns_prediction_result(fitted_classifier: SwahiliNewsClassifier) -> None:
    result = fitted_classifier.predict_proba("Timu imeshinda mechi kubwa")
    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    assert result.label in result.probabilities


def test_predict_batch(fitted_classifier: SwahiliNewsClassifier) -> None:
    preds = fitted_classifier.predict_batch(["Rais ametangaza", "Timu imeshinda"])
    assert len(preds) == 2


def test_save_and_load_roundtrip(fitted_classifier: SwahiliNewsClassifier) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.joblib"
        fitted_classifier.save(model_path)
        assert model_path.exists()

        reloaded = SwahiliNewsClassifier.load(model_path)
        original_pred = fitted_classifier.predict("Bunge limepitisha sheria")
        reloaded_pred = reloaded.predict("Bunge limepitisha sheria")
        assert original_pred == reloaded_pred


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        SwahiliNewsClassifier.load("/nonexistent/path/model.joblib")
