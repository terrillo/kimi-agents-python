"""Pre-upload format checks — offline, no API call needed."""

from kimi_agents_python import (
    FILE_EXTRACT_FORMATS,
    IMAGE_FORMATS,
    VIDEO_FORMATS,
    FilePurpose,
    supported_formats,
    validate_file_format,
)

print("Image formats:", sorted(IMAGE_FORMATS))
print("Video formats:", sorted(VIDEO_FORMATS))
print(f"file-extract accepts {len(FILE_EXTRACT_FORMATS)} formats\n")

validate_file_format("photo.png", FilePurpose.IMAGE)
print("photo.png is valid for purpose=image ✓")

try:
    validate_file_format("clip.mp4", FilePurpose.IMAGE)
except ValueError as e:
    print(f"\nclip.mp4 for purpose=image:\n  {e}")

print(f"\nfile-extract sample: {sorted(supported_formats('file-extract'))[:10]} ...")
