"""
Curated list of common Swahili stop words.

Swahili does not have a single authoritative stop-word list the way NLTK
ships one for English, so this set was compiled from high-frequency
function words: pronouns, conjunctions, prepositions, common verb
particles ("ni", "si", "ya", "wa" agreement markers) and discourse
markers that carry little topical meaning for a bag-of-words model.

The list is intentionally conservative -- it excludes words that are
frequently ambiguous between "stop word" and "content word" in short
headlines (e.g. "wa" can be a possessive marker or part of "wananchi").
Extend or trim ``SWAHILI_STOPWORDS`` for your own corpus as needed.
"""

from __future__ import annotations

SWAHILI_STOPWORDS: frozenset[str] = frozenset(
    {
       "na", "ya", "wa", "za", "la", "cha", "vya", "kwa", "katika",
        "kuwa", "ni", "si", "au", "lakini", "kama", "hii", "hiyo",
        "hilo", "huu", "huo", "wale", "hawa", "hao", "yeye", "wao",
        "sisi", "wewe", "mimi", "nyinyi", "yake", "yangu", "yetu",
        "yenu", "wangu", "wako", "wetu", "wenu", "wao", "ndani",
        "nje", "juu", "chini", "baada", "kabla", "wakati", "leo",
        "jana", "kesho", "sasa", "tayari", "bado", "tena", "pia",
        "hata", "ili", "kwamba", "ambao", "ambayo", "ambaye",
        "ambacho", "ambavyo", "kila", "wote", "zote", "vyote",
        "yote", "hakuna", "kuna", "kutoka", "hadi", "mpaka",
        "zaidi", "sana", "tu", "je", "ndiyo", "hapana", "bila",
        "kwenye", "kutokana", "kuhusu", "miongoni", "mwa", "mno",
        "au", "ingawa", "ijapokuwa", "licha", "kwani", "kwa sababu",
        "hivyo", "basi", "tokea", "toka", "gani", "vipi", "wapi",
        "lini", "nani", "nini", "vile", "hivi", "vingi", "kidogo",
        "moja", "mbili", "tatu", "the", "a", "an", "is", "are", 
    }
)