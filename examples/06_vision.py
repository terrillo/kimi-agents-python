"""Vision — describe an image. Usage: uv run python examples/06_vision.py photo.png"""

import base64
import sys
from pathlib import Path

from kimi_agents_python import FilePurpose, KimiClient, Model, validate_file


def encode_image(path: Path) -> str:
    validate_file(path, FilePurpose.IMAGE)
    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext == "jpg" else ext
    return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def main(image_path: str) -> None:
    url = encode_image(Path(image_path))
    with KimiClient() as client:
        response = client.chat.create(
            model=Model.KIMI_K2_6,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": "Describe what's in this image."},
                    ],
                }
            ],
            max_tokens=400,
        )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python 06_vision.py <path/to/image.png>")
    main(sys.argv[1])
