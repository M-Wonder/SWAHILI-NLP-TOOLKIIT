"""
Feature extraction utilities that turn preprocessed Swahili text into
numeric vectors suitable for scikit-learn estimators.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import tfidf_vectorizer

from swahili_nlp.preprocessing import SwahiliTextPreprocessor


def build_tfidf_vectorizer(
         preprocessor: SwahiliTextPreprocessor | None = None,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Construct a :class:`TfidfVectorizer` wired to Swahili preprocessing.

    Parameters
    ----------
    preprocessor:
        The :class:`SwahiliTextPreprocessor` used to clean and tokenize
        text before vectorization. A default instance is created if not
        supplied.
    max_features:
        Maximum vocabulary size kept by TF-IDF (most frequent terms).
    ngram_range:
        Range of n-gram sizes to extract; ``(1, 2)`` captures unigrams
        and bigrams, which helps with short headline-style text.

    Returns
    -------
    TfidfVectorizer
        An unfitted vectorizer ready to be used in a pipeline or fitted
        directly via ``.fit_transform``.
    """
    pre = preprocessor or SwahiliTextPreprocessor()
    return TfidfVectorizer(
        tokenizer=pre.tokenize,
        preprocessor=pre.normalize,
        token_pattern=None,  # disabled because a custom tokenizer is supplied
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )

        
