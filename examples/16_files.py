"""Files API — upload, list, retrieve content, delete."""

from pathlib import Path

from kimi_agents_python import FilePurpose, KimiClient, Model

# Write a tiny sample so the example is self-contained.
sample = Path("sample.md")
sample.write_text("# Demo\n\nKimi files API can extract text from this.\n")

with KimiClient() as client:
    print("=== Upload ===")
    f = client.files.create(file=sample, purpose=FilePurpose.FILE_EXTRACT)
    print(f"  {f.id}  {f.filename}  {f.bytes}B  status={f.status}")

    print("\n=== Extracted content ===")
    body = client.files.content(f.id)
    print(body.text)

    print("=== Use it in a chat ===")
    response = client.chat.create(
        model=Model.KIMI_K2_0905_PREVIEW,
        messages=[
            {"role": "system", "content": body.text},
            {"role": "user", "content": "Summarize this file in one sentence."},
        ],
        max_tokens=200,
    )
    print(response.choices[0].message.content)

    print("\n=== Cleanup ===")
    deleted = client.files.delete(f.id)
    print(f"  deleted={deleted.deleted}")

sample.unlink(missing_ok=True)
