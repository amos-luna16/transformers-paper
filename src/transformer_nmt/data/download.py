from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import IO

from transformer_nmt.config import DataSettings

logger = logging.getLogger(__name__)

_USER_AGENT = "transformer-nmt/0.1 (+https://github.com/Amos-Luna/transformers-paper)"


def _write_atomically(destination: Path, write: Callable[[IO[bytes]], None]) -> Path:
    """Stream into a temporary file next to ``destination`` and rename it into place on success.

    Args:
        destination: Final file path; parent directories are created.
        write: Callback that writes the payload into the open temporary file.

    Returns:
        The destination path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=destination.parent, delete=False)
    temporary_path = Path(handle.name)
    try:
        with handle:
            write(handle)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def download_file(url: str, destination: Path, *, timeout_seconds: int = 120) -> Path:
    """Download ``url`` to ``destination`` atomically.

    Args:
        url: HTTP(S) URL to fetch.
        destination: Target file path; parent directories are created.
        timeout_seconds: Socket timeout for the request.

    Returns:
        The destination path.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    def fetch(handle: IO[bytes]) -> None:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            shutil.copyfileobj(response, handle)

    _write_atomically(destination, fetch)
    logger.info(
        "Downloaded %s -> %s (%.1f MB)", url, destination, destination.stat().st_size / 1e6
    )
    return destination


def extract_archive_member(archive_path: Path, member: str, destination: Path) -> Path:
    """Extract a single member of a zip archive to ``destination`` atomically.

    Args:
        archive_path: Zip file on disk.
        member: Path of the file inside the archive.
        destination: Where the extracted file is written.

    Returns:
        The destination path.
    """

    def extract(handle: IO[bytes]) -> None:
        with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source:
            shutil.copyfileobj(source, handle)

    return _write_atomically(destination, extract)


def ensure_dataset_file(settings: DataSettings, data_dir: Path) -> Path:
    """Return the path of the sentence-pairs TSV, downloading and extracting it if needed.

    Args:
        settings: Data settings with the download URL and archive member name.
        data_dir: Directory where raw files are cached.

    Returns:
        Path to the tab-separated pairs file.
    """
    if settings.local_file is not None:
        if not settings.local_file.exists():
            raise FileNotFoundError(f"DATA_LOCAL_FILE not found: {settings.local_file}")
        return settings.local_file

    pairs_file = data_dir / Path(settings.archive_member).name
    if pairs_file.exists():
        return pairs_file

    archive_path = data_dir / Path(settings.source_url).name
    if not archive_path.exists():
        download_file(settings.source_url, archive_path)
    logger.info("Extracting %s from %s", settings.archive_member, archive_path)
    return extract_archive_member(archive_path, settings.archive_member, pairs_file)
