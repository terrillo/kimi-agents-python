from __future__ import annotations

from pathlib import Path

from ._enums import FilePurpose

# Per https://platform.kimi.ai/docs/guide/use-kimi-vision-model — Supported Formats.
IMAGE_FORMATS: frozenset[str] = frozenset({"png", "jpeg", "jpg", "webp", "gif"})
VIDEO_FORMATS: frozenset[str] = frozenset(
    {"mp4", "mpeg", "mpg", "mov", "avi", "x-flv", "flv", "webm", "wmv", "3gpp", "3gp"}
)

# Per https://platform.kimi.ai/docs/api/files-upload — Supported Formats accordion.
FILE_EXTRACT_FORMATS: frozenset[str] = frozenset(
    {
        "pdf", "txt", "csv", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "md",
        "jpeg", "jpg", "png", "bmp", "gif", "svg", "svgz", "webp", "ico", "xbm",
        "dib", "pjp", "tif", "pjpeg", "avif", "dot", "apng", "epub", "tiff",
        "jfif", "html", "json", "mobi", "log",
        "go", "h", "c", "cpp", "cxx", "cc", "cs", "java", "js", "css",
        "jsp", "php", "py", "py3", "asp", "yaml", "yml", "ini", "conf",
        "ts", "tsx",
    }
)

BATCH_FORMATS: frozenset[str] = frozenset({"jsonl"})

MAX_FILE_BYTES: int = 100 * 1024 * 1024  # 100 MiB per file
MAX_TOTAL_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GiB total quota
MAX_FILES: int = 1_000

_FORMATS_BY_PURPOSE: dict[FilePurpose, frozenset[str]] = {
    FilePurpose.IMAGE: IMAGE_FORMATS,
    FilePurpose.VIDEO: VIDEO_FORMATS,
    FilePurpose.FILE_EXTRACT: FILE_EXTRACT_FORMATS,
    FilePurpose.BATCH: BATCH_FORMATS,
}


def _as_purpose(purpose: FilePurpose | str) -> FilePurpose:
    return purpose if isinstance(purpose, FilePurpose) else FilePurpose(purpose)


def supported_formats(purpose: FilePurpose | str) -> frozenset[str]:
    """Return the set of allowed file extensions for ``purpose`` (without the dot)."""
    return _FORMATS_BY_PURPOSE[_as_purpose(purpose)]


def file_extension(path: str | Path) -> str:
    """Lowercase extension without the leading dot."""
    return Path(path).suffix.lstrip(".").lower()


def validate_file_format(path: str | Path, purpose: FilePurpose | str) -> None:
    """Raise ValueError if ``path``'s extension isn't supported for ``purpose``."""
    key = _as_purpose(purpose)
    ext = file_extension(path)
    allowed = _FORMATS_BY_PURPOSE[key]
    if ext not in allowed:
        raise ValueError(
            f"File extension '.{ext}' is not supported for purpose='{key.value}'. "
            f"Allowed: {sorted(allowed)}"
        )


def validate_file_size(path: str | Path) -> None:
    """Raise ValueError if the file is empty or exceeds the per-file upload limit."""
    p = Path(path)
    size = p.stat().st_size
    if size == 0:
        raise ValueError(f"File {p} is empty (zero bytes).")
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"File {p} is {size} bytes; max upload size is "
            f"{MAX_FILE_BYTES} bytes (100 MiB)."
        )


def validate_file(path: str | Path, purpose: FilePurpose | str) -> None:
    """Run both format and size validation. Raises ValueError on any violation."""
    p = Path(path)
    validate_file_format(p, purpose)
    validate_file_size(p)
