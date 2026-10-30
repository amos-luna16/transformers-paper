from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transformer_nmt.config import get_settings
from transformer_nmt.inference.translator import Translator
from transformer_nmt.utils.hardware import resolve_device
from transformer_nmt.utils.logging import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse inference options."""
    parser = argparse.ArgumentParser(
        description="Translate English sentences to French with a trained checkpoint."
    )
    parser.add_argument(
        "sentences", nargs="*", help="Sentences to translate; omit for interactive mode."
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Path to best.pt/last.pt; defaults to the best run on disk."
    )
    parser.add_argument("--beam-size", type=int, help="1 = greedy decoding; >1 = beam search.")
    parser.add_argument(
        "--length-penalty", type=float, help="Beam-search length-normalisation exponent."
    )
    parser.add_argument("--device", type=str, help="cpu, mps or cuda (auto-detected by default).")
    return parser.parse_args(argv)


def load_translator(args: argparse.Namespace) -> Translator:
    """Instantiate the translator from CLI arguments and settings."""
    settings = get_settings()
    device = resolve_device(args.device)
    checkpoint = args.checkpoint or settings.inference.checkpoint_path
    if checkpoint is not None:
        return Translator.from_checkpoint(
            checkpoint, device, max_decode_len=settings.inference.max_decode_len
        )
    return Translator.from_best_run(
        settings.tracking.artifacts_dir, device, max_decode_len=settings.inference.max_decode_len
    )


def main(argv: list[str] | None = None) -> int:
    """Translate the given sentences or start an interactive prompt."""
    configure_logging(level=logging.INFO)
    args = parse_args(argv)
    settings = get_settings()
    beam_size = args.beam_size if args.beam_size is not None else settings.inference.beam_size
    length_penalty = (
        args.length_penalty
        if args.length_penalty is not None
        else settings.inference.length_penalty
    )
    translator = load_translator(args)

    if args.sentences:
        for sentence in args.sentences:
            translation = translator.translate(
                sentence, beam_size=beam_size, length_penalty=length_penalty
            )
            print(f"EN: {sentence}\nFR: {translation}\n")
        return 0

    print("Interactive mode - type an English sentence (empty line to quit).")
    while True:
        try:
            sentence = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not sentence:
            break
        print(
            f"FR> {translator.translate(sentence, beam_size=beam_size, length_penalty=length_penalty)}"  # noqa: E501
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
