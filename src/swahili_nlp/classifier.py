"""
Swahili news headline classifier.

Wraps a scikit-learn Pipeline (TF-IDF -> Logistic Regression) behind a
small, typed API that is convenient to train, evaluate, persist, and
reload -- the kind of interface a downstream service or CLI would call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from swahili_nlp.features import build_tfidf_vectorizer
from swahili_nlp.preprocessing import SwahiliTextPreprocessor

logger = logging.getLogger(__name__)


class NotFittedError(RuntimeError):
    """Raised when a prediction is requested from an untrained classifier."""


@dataclass
class PredictionResult:
    """Result of a single classification call.

    Attributes
    ----------
    text:
        The original input text.
    label:
        The predicted category label.
    confidence:
        Probability assigned to the predicted label by the model.
    probabilities:
        Full probability distribution over all known categories.
    """

    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]


class SwahiliNewsClassifier:
    """TF-IDF + Logistic Regression classifier for Swahili news headlines.

    Examples
    --------
    >>> clf = SwahiliNewsClassifier()
    >>> clf.fit(["Rais ametangaza sera mpya", "Timu imeshinda mechi"],
    ...         ["siasa", "michezo"])  # doctest: +SKIP
    >>> clf.predict("Bunge limepitisha sheria mpya")  # doctest: +SKIP
    'siasa'
    """

    def __init__(
        self,
        preprocessor: SwahiliTextPreprocessor | None = None,
        max_features: int = 5000,
        C: float = 5.0,
        random_state: int = 42,
    ) -> None:
        self.preprocessor = preprocessor or SwahiliTextPreprocessor()
        vectorizer = build_tfidf_vectorizer(
            self.preprocessor, max_features=max_features
        )
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("tfidf", vectorizer),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=1000,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self._is_fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> "SwahiliNewsClassifier":
        """Train the pipeline on raw ``texts`` and their category ``labels``.

        Raises
        ------
        ValueError
            If ``texts`` and ``labels`` are empty or of mismatched length.
        """
        if not texts or not labels:
            raise ValueError("texts and labels must be non-empty")
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must be the same length "
                f"(got {len(texts)} and {len(labels)})"
            )
        logger.info("Training on %d examples across %d classes",
                     len(texts), len(set(labels)))
        self.pipeline.fit(texts, labels)
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(
                "Classifier has not been trained. Call .fit(...) or "
                ".load(...) before predicting."
            )

    def predict(self, text: str) -> str:
        """Return the single most likely category for ``text``."""
        self._check_fitted()
        return self.pipeline.predict([text])[0]

    def predict_proba(self, text: str) -> PredictionResult:
        """Return the full :class:`PredictionResult` (label + probabilities)."""
        self._check_fitted()
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        prob_map = {cls: float(p) for cls, p in zip(classes, proba)}
        best_idx = proba.argmax()
        return PredictionResult(
            text=text,
            label=classes[best_idx],
            confidence=float(proba[best_idx]),
            probabilities=prob_map,
        )

    def predict_batch(self, texts: list[str]) -> list[str]:
        """Predict categories for a batch of texts."""
        self._check_fitted()
        return list(self.pipeline.predict(texts))

    def save(self, path: str | Path) -> None:
        """Persist the fitted pipeline to ``path`` using joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SwahiliNewsClassifier":
        """Load a previously saved pipeline from ``path``.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved model found at {path}")
        instance = cls()
        instance.pipeline = joblib.load(path)
        instance._is_fitted = True
        logger.info("Model loaded from %s", path)
        return instance
