"""
swahili_nlp
===========

A lightweight NLP toolkit for preprocessing and classifying Swahili text.

Public API
----------
- SwahiliTextPreprocessor : text normalization, tokenization, stop-word removal
- SwahiliNewsClassifier   : TF-IDF + Logistic Regression pipeline for headline
                            category classification
- SWAHILI_STOPWORDS       : curated Swahili stop-word set
"""

from swahili_nlp.preprocessing import SwahiliTextPreprocessor
from swahili_nlp.classifier import SwahiliNewsClassifier
from swahili_nlp.stopwords import SWAHILI_STOPWORDS

__version__ = "1.0.0"
__all__ = [
    "SwahiliTextPreprocessor",
    "SwahiliNewsClassifier",
    "SWAHILI_STOPWORDS",
]
