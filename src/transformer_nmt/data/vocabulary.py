from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from transformer_nmt.config import (
    EOS_INDEX,
    PAD_INDEX,
    SOS_INDEX,
    SPECIAL_TOKENS,
    UNK_INDEX,
)


class Vocabulary:
    """Bidirectional token/index mapping with the four reserved special tokens first."""

    def __init__(
        self,
        tokens: Sequence[str],
    ) -> None:
        """Create a vocabulary from an ordered token list.

        Args:
            tokens: Unique tokens whose first four entries must be the special tokens.

        Raises:
            ValueError: If the special tokens are missing, misplaced or tokens repeat.
        """
        if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"Vocabulary must start with the special tokens {SPECIAL_TOKENS}")
        if len(set(tokens)) != len(tokens):
            raise ValueError("Vocabulary tokens must be unique")
        self._tokens: list[str] = list(tokens)
        self._index_by_token: dict[str, int] = {tok: i for i, tok in enumerate(self._tokens)}

    @classmethod
    def build(
        cls,
        sentences: Iterable[str],
        tokenizer: Callable[[str], list[str]],
        min_freq: int = 1,
    ) -> Vocabulary:
        """Build a vocabulary from a corpus, keeping tokens seen at least ``min_freq`` times.

        Tokens are ordered by descending frequency, then alphabetically, so the result is
        deterministic for a given corpus.

        Args:
            sentences: Corpus sentences.
            tokenizer: Callable splitting a sentence into tokens.
            min_freq: Minimum number of occurrences required.

        Returns:
            The constructed vocabulary.
        """
        counter: Counter[str] = Counter()
        for sentence in sentences:
            counter.update(tokenizer(sentence))
        kept = sorted(
            (
                tok
                for tok, count in counter.items()
                if count >= min_freq and tok not in SPECIAL_TOKENS
            ),
            key=lambda tok: (-counter[tok], tok),
        )
        return cls([*SPECIAL_TOKENS, *kept])

    @property
    def pad_index(self) -> int:
        """Index of the padding token."""
        return PAD_INDEX

    @property
    def unk_index(self) -> int:
        """Index of the unknown-word token."""
        return UNK_INDEX

    @property
    def sos_index(self) -> int:
        """Index of the start-of-sequence token."""
        return SOS_INDEX

    @property
    def eos_index(self) -> int:
        """Index of the end-of-sequence token."""
        return EOS_INDEX

    @property
    def tokens(self) -> list[str]:
        """Copy of the ordered token list."""
        return list(self._tokens)

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self._tokens)

    def __contains__(
        self,
        token: str,
    ) -> bool:
        """Return whether ``token`` has a dedicated index."""
        return token in self._index_by_token

    def token_to_index(
        self,
        token: str,
    ) -> int:
        """Map a token to its index, falling back to ``<unk>``."""
        return self._index_by_token.get(token, UNK_INDEX)

    def index_to_token(
        self,
        index: int,
    ) -> str:
        """Map an index back to its token."""
        return self._tokens[index]

    def encode(
        self,
        tokens: Sequence[str],
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """Convert tokens into indices, optionally wrapping them in ``<sos>``/``<eos>``.

        Args:
            tokens: Tokens to encode.
            add_special_tokens: Prepend ``<sos>`` and append ``<eos>`` when True.

        Returns:
            List of token indices.
        """
        indices = [self.token_to_index(tok) for tok in tokens]
        if add_special_tokens:
            return [SOS_INDEX, *indices, EOS_INDEX]
        return indices

    def decode(
        self,
        indices: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Convert indices back into tokens, stopping at the first ``<eos>``.

        Args:
            indices: Token indices to decode.
            skip_special_tokens: Drop ``<pad>``, ``<sos>`` and ``<eos>`` from the output.

        Returns:
            List of tokens.
        """
        tokens: list[str] = []
        for index in indices:
            if skip_special_tokens and index == EOS_INDEX:
                break
            if skip_special_tokens and index in (PAD_INDEX, SOS_INDEX):
                continue
            tokens.append(self._tokens[index])
        return tokens

    def save(
        self,
        path: Path,
    ) -> None:
        """Serialise the vocabulary to a JSON file.

        Args:
            path: Destination file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": self._tokens}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> Vocabulary:
        """Deserialise a vocabulary written by :meth:`save`.

        Args:
            path: JSON file to read.

        Returns:
            The restored vocabulary.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["tokens"])
