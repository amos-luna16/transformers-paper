from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_PUNCTUATION_PATTERN = re.compile(r"([.!?])")
_APOSTROPHE_PATTERN = re.compile(r"[\u2018\u2019\u02bc]")
# Latin-1 letters (excluding the × and ÷ signs) plus the French œ ligature.
_ALLOWED_CHARACTERS_PATTERN = re.compile(r"[^a-zA-ZÀ-ÖØ-öø-ÿŒœ'.!?]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_TOKEN_PATTERN = re.compile(r"\b\w+(?:'\w+)*\b|[.!?]")


def normalize_text(
    text: str,
) -> str:
    """Lower-case and clean a Latin-script sentence for word-level tokenisation.

    The function keeps accented letters, the ``œ`` ligature and apostrophes (contractions such
    as ``c'est``; typographic apostrophes are mapped to ASCII), separates sentence punctuation
    from words and collapses repeated whitespace.

    Args:
        text: Raw sentence.

    Returns:
        The normalised sentence.
    """
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = _APOSTROPHE_PATTERN.sub("'", text)
    text = _PUNCTUATION_PATTERN.sub(r" \1", text)
    text = _ALLOWED_CHARACTERS_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


class WordTokenizer:
    """Regex word-level tokenizer that keeps contractions and sentence punctuation."""

    def __call__(
        self,
        text: str,
    ) -> list[str]:
        """Split a sentence into lower-cased word and punctuation tokens.

        Args:
            text: Sentence to tokenise (normalised or raw).

        Returns:
            Ordered list of tokens.
        """
        return _TOKEN_PATTERN.findall(text.lower())

    def detokenize(
        self,
        tokens: Sequence[str],
    ) -> str:
        """Join tokens back into a readable sentence.

        Args:
            tokens: Tokens produced by :meth:`__call__` or by a model.

        Returns:
            A sentence with punctuation attached to the preceding word.
        """
        sentence = " ".join(tokens)
        return re.sub(r"\s+([.!?])", r"\1", sentence)
