"""Resume: submit batches 3 and 4, poll each to completion, then download all."""

import json
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OUTPUT_DIR = Path("data/labelled/topics")
WATCH_INTERVAL = 60

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load existing batch IDs
ids_path = OUTPUT_DIR / "batch_ids.txt"
batch_ids = [l.strip() for l in ids_path.read_text().splitlines() if l.strip()]
print(f"Existing batches: {len(batch_ids)}")
for i, bid in enumerate(batch_ids):
    b = client.batches.retrieve(bid)
    print(f"  Batch {i+1}: {b.status} ({b.request_counts.completed}/{b.request_counts.total})")

# Submit remaining JSONL files
remaining = []
for n in [4]:
    p = OUTPUT_DIR / f"topic_batch_requests_{n}.jsonl"
    if p.exists():
        remaining.append((n, p))

if not remaining:
    print("\nNo remaining JSONL files to submit.")
else:
    for chunk_num, jsonl_path in remaining:
        print(f"\n{'='*50}")
        print(f"Submitting batch {chunk_num} ({jsonl_path.name})")
        print(f"{'='*50}")

        # Count requests
        with open(jsonl_path, encoding="utf-8") as f:
            req_count = sum(1 for _ in f)

        # Submit with retry
        batch_obj = None
        for attempt in range(120):
            try:
                with open(jsonl_path, "rb") as f:
                    uploaded = client.files.create(file=f, purpose="batch")
                batch_obj = client.batches.create(
                    input_file_id=uploaded.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"description": f"DDDS topic extraction -- part {chunk_num}/4"},
                )
                print(f"Submitted: {batch_obj.id} ({req_count:,} requests)")
                break
            except Exception as e:
                if "enqueued" in str(e).lower() or "limit" in str(e).lower():
                    print(f"Enqueue limit hit, retrying in 60s... (attempt {attempt+1})")
                    time.sleep(60)
                else:
                    raise

        if not batch_obj:
            print(f"[!] Failed to submit batch {chunk_num}. Stopping.")
            break

        batch_ids.append(batch_obj.id)
        with open(ids_path, "w") as f:
            f.write("\n".join(batch_ids))

        # Poll until complete
        while True:
            batch_obj = client.batches.retrieve(batch_obj.id)
            completed = batch_obj.request_counts.completed
            total = batch_obj.request_counts.total
            failed = batch_obj.request_counts.failed
            pct = 100 * completed / max(total, 1)

            timestamp = datetime.now().strftime("%H:%M:%S")
            filled = int(20 * completed / max(total, 1))
            bar = "█" * filled + "░" * (20 - filled)

            line = f"[{timestamp}]  {bar}  {pct:5.1f}%  ({completed:,}/{total:,})"
            if failed:
                line += f"  [{failed:,} failed]"
            line += f"  ({batch_obj.status})"
            print(f"  {line}")

            if batch_obj.status == "completed":
                print(f"Batch {chunk_num} complete!")
                break
            if batch_obj.status in ("failed", "expired", "cancelled"):
                print(f"[!] Batch {chunk_num} {batch_obj.status}. Retrying...")
                # Resubmit
                for attempt in range(120):
                    try:
                        with open(jsonl_path, "rb") as f:
                            uploaded = client.files.create(file=f, purpose="batch")
                        batch_obj = client.batches.create(
                            input_file_id=uploaded.id,
                            endpoint="/v1/chat/completions",
                            completion_window="24h",
                            metadata={"description": f"DDDS topic extraction -- part {chunk_num}/4 (retry)"},
                        )
                        batch_ids[-1] = batch_obj.id
                        with open(ids_path, "w") as f:
                            f.write("\n".join(batch_ids))
                        print(f"Resubmitted: {batch_obj.id}")
                        break
                    except Exception as e:
                        if "enqueued" in str(e).lower() or "limit" in str(e).lower():
                            print(f"Enqueue limit hit, retrying in 60s... (attempt {attempt+1})")
                            time.sleep(60)
                        else:
                            raise
                continue

            time.sleep(WATCH_INTERVAL)

print(f"\nAll batches done. Now run:")
print(f'  python "Pipeline/run 1 - Topic Extraction.py" --mode download')