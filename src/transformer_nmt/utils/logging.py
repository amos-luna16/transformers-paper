from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | rank%(rank)s | %(name)s | %(message)s"


class _RankFilter(logging.Filter):
    """Inject the process rank into every log record."""

    def __init__(self, rank: int) -> None:
        """Store the rank to inject."""
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach ``rank`` to the record and keep it."""
        record.rank = self.rank
        return True


def configure_logging(
    rank: int = 0, *, level: int = logging.INFO, quiet_non_main: bool = True
) -> None:
    """Configure root logging once per process with a rank-aware format.

    Args:
        rank: Global rank of the process (0 for single-process runs).
        level: Log level for the main process.
        quiet_non_main: Raise non-main ranks to ``WARNING`` to avoid duplicated lines.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    handler.addFilter(_RankFilter(rank))
    root.addHandler(handler)
    root.setLevel(level if rank == 0 or not quiet_non_main else logging.WARNING)
    for noisy in ("urllib3", "matplotlib", "PIL", "git", "mlflow.utils", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
