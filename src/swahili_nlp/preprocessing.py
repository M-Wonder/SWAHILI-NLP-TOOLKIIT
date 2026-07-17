"""
Text preprocessing utilities for Swahili text.

This module implements a small, dependency-free preprocessing pipeline
(normalization -> tokenization -> stop-word removal) that is used both
by the CLI and by the feature extraction layer in :mod:`swahili_nlp.features`.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

from swahili_nlp.stopwords import SWAHILI_STOPWORDS

logger = logging.getLogger(__name__)

# Matches runs of Swahili/Latin alphabetic characters. Swahili orthography
# uses the standard Latin alphabet (no diacritics), so a simple \w-based
# pattern is sufficient and avoids pulling in a heavier tokenizer library.
_TOKEN_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ]+")


@dataclass
class SwahiliTextPreprocessor:
    """Normalize and tokenize Swahili text for downstream NLP tasks.

    Parameters
    ----------
    remove_stopwords:
        If ``True``, tokens present in :data:`SWAHILI_STOPWORDS` are dropped.
    min_token_length:
        Tokens shorter than this length are discarded (default 2, which
        filters out single-letter noise while keeping short valid words
        such as "na" if stop-word removal is disabled).
    lowercase:
        If ``True``, text is lowercased before tokenization.

    Examples
    --------
    >>> pre = SwahiliTextPreprocessor()
    >>> pre.tokenize("Rais ametangaza bajeti mpya ya Serikali!")
    ['rais', 'ametangaza', 'bajeti', 'mpya', 'serikali']
    """
    remove_stopwords: bool = True
    min_token_length: int = 2
    lowercase: bool = True
    stopwords: frozenset[str] = field(default_factory=lambda: frozenset(SWAHILI_STOPWORDS)
    )

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        normalized = text.strip()
        if self.lowercase:
            normalized = normalized.lower()
        return normalized
    
    def tokenize(self, text: str) -> list[str]:
        """Normalize, tokenize, and filter ``text`` into a list of tokens.

        Punctuation and digits are dropped, tokens shorter than
        ``min_token_length`` are removed, and stop words are removed if
        ``remove_stopwords`` is ``True``.
        """
        normalized = self.normalize(text)
        tokens = _TOKEN_PATTERN.findall(normalized)
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens
    
    def preprocess(self, text):
        tokens = self.tokenize(text)
        return " ".join(tokens)
    
    def preprocess_batch(self, texts):
        return [self.preprocess(text) for text in texts]



