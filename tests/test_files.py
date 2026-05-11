from __future__ import annotations

from pathlib import Path

import pytest

from kimi_agents_python import (
    BATCH_FORMATS,
    FILE_EXTRACT_FORMATS,
    IMAGE_FORMATS,
    MAX_FILE_BYTES,
    VIDEO_FORMATS,
    FilePurpose,
    file_extension,
    supported_formats,
    validate_file,
    validate_file_format,
    validate_file_size,
)


def test_image_formats_match_docs() -> None:
    # Per https://platform.kimi.ai/docs/guide/use-kimi-vision-model
    for ext in ("png", "jpeg", "webp", "gif"):
        assert ext in IMAGE_FORMATS


def test_video_formats_match_docs() -> None:
    for ext in ("mp4", "mpeg", "mov", "avi", "x-flv", "mpg", "webm", "wmv", "3gpp"):
        assert ext in VIDEO_FORMATS


def test_file_extract_formats_include_common_types() -> None:
    for ext in ("pdf", "docx", "txt", "csv", "md", "html", "py", "json"):
        assert ext in FILE_EXTRACT_FORMATS


def test_batch_formats_is_jsonl_only() -> None:
    assert BATCH_FORMATS == frozenset({"jsonl"})


def test_supported_formats_by_purpose() -> None:
    assert supported_formats(FilePurpose.IMAGE) is IMAGE_FORMATS
    assert supported_formats(FilePurpose.VIDEO) is VIDEO_FORMATS
    assert supported_formats(FilePurpose.FILE_EXTRACT) is FILE_EXTRACT_FORMATS
    assert supported_formats(FilePurpose.BATCH) is BATCH_FORMATS
    # Accepts the raw string too.
    assert supported_formats("image") is IMAGE_FORMATS


def test_supported_formats_rejects_unknown_purpose() -> None:
    with pytest.raises(ValueError):
        supported_formats("not-a-real-purpose")


@pytest.mark.parametrize(
    "path, expected",
    [
        ("photo.PNG", "png"),
        ("/abs/path/clip.MP4", "mp4"),
        ("no-extension", ""),
        ("archive.tar.gz", "gz"),
    ],
)
def test_file_extension(path: str, expected: str) -> None:
    assert file_extension(path) == expected


def test_validate_file_format_accepts_supported(tmp_path: Path) -> None:
    f = tmp_path / "img.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    validate_file_format(f, FilePurpose.IMAGE)


def test_validate_file_format_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="not supported for purpose='image'"):
        validate_file_format("clip.mp4", FilePurpose.IMAGE)
    with pytest.raises(ValueError, match="not supported for purpose='video'"):
        validate_file_format("doc.pdf", FilePurpose.VIDEO)


def test_validate_file_format_accepts_string_purpose() -> None:
    validate_file_format("photo.jpeg", "image")


def test_validate_file_size_empty_raises(tmp_path: Path) -> None:
    f = tmp_path / "empty.png"
    f.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        validate_file_size(f)


def test_validate_file_size_too_large_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "big.png"
    f.write_bytes(b"x" * 10)
    monkeypatch.setattr("kimi_agents_python.files.MAX_FILE_BYTES", 5)
    with pytest.raises(ValueError, match="max upload size"):
        validate_file_size(f)


def test_validate_file_size_ok(tmp_path: Path) -> None:
    f = tmp_path / "ok.png"
    f.write_bytes(b"x" * 32)
    validate_file_size(f)


def test_validate_file_combined(tmp_path: Path) -> None:
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x" * 16)
    validate_file(f, FilePurpose.VIDEO)

    bad = tmp_path / "clip.exe"
    bad.write_bytes(b"x" * 16)
    with pytest.raises(ValueError, match="not supported"):
        validate_file(bad, FilePurpose.VIDEO)


def test_max_file_bytes_constant() -> None:
    assert MAX_FILE_BYTES == 100 * 1024 * 1024


def test_file_purpose_values() -> None:
    assert FilePurpose.FILE_EXTRACT == "file-extract"
    assert FilePurpose.IMAGE == "image"
    assert FilePurpose.VIDEO == "video"
    assert FilePurpose.BATCH == "batch"
