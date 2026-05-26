"""Batches API — submit a JSONL job, poll for status, fetch results."""

import json
import time
from pathlib import Path

from kimi_agents_python import BatchStatus, FilePurpose, KimiClient

JOBS = [
    {"role": "user", "content": "Capital of France?"},
    {"role": "user", "content": "Capital of Japan?"},
]

batch_path = Path("batch_input.jsonl")
with batch_path.open("w") as fh:
    for i, msg in enumerate(JOBS):
        line = {
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "kimi-k2.6", "messages": [msg]},
        }
        fh.write(json.dumps(line) + "\n")

with KimiClient() as client:
    print("=== Upload input ===")
    input_file = client.files.create(file=batch_path, purpose=FilePurpose.BATCH)
    print(f"  {input_file.id}")

    print("\n=== Submit batch ===")
    batch = client.batches.create(
        input_file_id=input_file.id,
        completion_window="24h",
        metadata={"demo": "capitals"},
    )
    print(f"  {batch.id}  status={batch.status}")

    print("\n=== Poll ===")
    terminal = {
        BatchStatus.COMPLETED,
        BatchStatus.FAILED,
        BatchStatus.EXPIRED,
        BatchStatus.CANCELLED,
    }
    MAX_WAIT_SECONDS = 600
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while True:
        batch = client.batches.retrieve(batch.id)
        print(
            f"  status={batch.status}  "
            f"counts={batch.request_counts.completed}/{batch.request_counts.total}"
            if batch.request_counts
            else f"  status={batch.status}"
        )
        if batch.status in terminal:
            break
        if time.monotonic() >= deadline:
            print(f"  giving up after {MAX_WAIT_SECONDS}s — batch still running")
            break
        time.sleep(5)

    if batch.status is BatchStatus.COMPLETED and batch.output_file_id:
        print("\n=== Results ===")
        output = client.files.content(batch.output_file_id)
        for line in output.text.splitlines():
            print(f"  {line}")

batch_path.unlink(missing_ok=True)
