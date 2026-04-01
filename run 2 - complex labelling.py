"""
DDDS - GPT-4o Vagueness Labelling Pipeline (Batch API)
=======================================================
Task:   Label disclosure sentences as SPECIFIC (0) or VAGUE (1)
Model:  GPT-4o via OpenAI Batch API (50% cheaper, ~1-3 hour turnaround)

What this script does
---------------------
1. Reads the training sentences CSV (from run_0)
2. Batches sentences into groups of 5 per API request
3. Uploads two identical batch jobs to OpenAI (for consensus labelling)
4. Polls until both batches complete (~1-3 hours)
5. Downloads results, compares consensus across the two runs
6. Exports three CSVs matching the format run_3 and run_4 expect:
     - labelled.csv         : all sentences with consensus labels
     - validation_subset.csv: 20% sample for human review
     - needs_review.csv     : sentences where the two runs disagreed

Batch API benefits
------------------
- 50% cheaper than synchronous API calls
- No rate limit issues — upload everything at once
- Typical turnaround 1-3 hours (24h SLA)

You can safely close the terminal while batches are processing.
Re-run the script and it will check status and resume automatically.

Usage
-----
    # First run — prepares and uploads batches
    python run_2_-_vague_labelling.py --input data/training_sentences/training_sentences.csv

    # Check status while waiting
    python run_2_-_vague_labelling.py --input data/training_sentences/training_sentences.csv

    # Force re-download and reprocess completed batches
    python run_2_-_vague_labelling.py --input data/training_sentences/training_sentences.csv --reprocess

    # Use cached results (skip all API calls)
    python run_2_-_vague_labelling.py --use-cached
"""

import argparse
import json
import os
import time
import sys

import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL           = "gpt-4o"
INPUT_CSV       = "data/training_sentences/training_sentences.csv"
OUTPUT_DIR      = "data/labelled/vagueness"
STATE_FILE      = "data/labelled/vagueness/batch_state.json"
BATCH_SIZE      = 5          # sentences per API request
VALIDATION_FRAC = 0.20       # fraction exported for human validation
POLL_INTERVAL   = 60         # seconds between status checks
SEED            = 42
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert financial analyst specialising in SEC disclosure quality assessment.

Your task is to classify disclosure sentences from company filings (10-K, 10-Q) as either SPECIFIC or VAGUE, following the Bushee, Gow, and Taylor (2018) framework for deliberate linguistic vagueness.

DEFINITIONS
-----------
VAGUE (1): Language that is deliberately non-committal, evasive, or imprecise in a way that obscures information about the company's financial position, risks, or outlook. Vague language avoids binding the company to concrete statements.

SPECIFIC (0): Language that provides concrete, verifiable information. This includes specific figures, defined timelines, named counterparties, quantified risks, or clear causal explanations.

VAGUENESS INDICATORS - language is likely VAGUE if it:
- Uses hedging phrases with no substantive content: "may", "might", "could potentially", "there can be no assurance", "we cannot predict"
- Replaces specific figures with qualitative descriptions: "significant", "material", "substantial", "considerable" without quantification
- Uses passive constructions that obscure agency: "it has been determined", "steps are being taken"
- Refers to unnamed risks or factors: "various factors", "certain risks", "unforeseen circumstances"
- Describes outcomes without causal mechanisms: "results may differ materially" with no explanation of how
- Uses boilerplate legal language that carries no company-specific information

SPECIFICITY INDICATORS - language is likely SPECIFIC if it:
- Provides quantified figures: percentages, dollar amounts, timeframes, unit counts
- Names specific counterparties, geographies, products, or contracts
- Describes concrete causal mechanisms: "because of X, we expect Y"
- Gives defined conditions under which outcomes change
- Cites specific contractual obligations or regulatory requirements

IMPORTANT DISTINCTION
---------------------
Do not confuse vagueness with complexity. Technically complex language (e.g. detailed accounting policy descriptions, engineering specifications) can be highly specific. Only label as VAGUE if the language is evasive or non-committal, not merely difficult to understand.

EXAMPLES
--------
VAGUE: "We may be subject to various risks that could materially affect our business, financial condition, and results of operations."
Reason: No specific risks named, no quantification, pure boilerplate hedge.

VAGUE: "Management is taking appropriate steps to address the situation."
Reason: No specifics on what steps, who is responsible, or what the situation is.

SPECIFIC: "Our pension obligation increased by $47.3 million in Q3 2024 due to a 0.4 percentage point decline in the discount rate applied to our US defined benefit plan."
Reason: Named figure, named cause, named plan, quantified driver.

SPECIFIC: "We have a $200 million revolving credit facility maturing in March 2027, of which $85 million was drawn as of the filing date."
Reason: Specific amount, specific instrument, specific maturity, specific utilisation.

BORDERLINE GUIDANCE
-------------------
If a sentence mixes specific and vague elements, classify based on the dominant character. If a sentence contains one specific figure embedded in otherwise entirely evasive language, lean VAGUE. If a sentence is technically hedged but provides substantive specific context, lean SPECIFIC.

OUTPUT FORMAT
-------------
You will receive multiple numbered sentences. Classify each one independently.

Respond ONLY with a valid JSON object containing a "results" array. Each element must have the sentence number, label, confidence, and reason. Example:

{
  "results": [
    {"sentence": 1, "label": 0, "confidence": "high", "reason": "Contains specific dollar amounts and named facility"},
    {"sentence": 2, "label": 1, "confidence": "high", "reason": "Pure boilerplate hedge with no specific content"},
    {"sentence": 3, "label": 0, "confidence": "medium", "reason": "Names specific geography and product line"}
  ]
}"""


def build_user_prompt(sentences: list[dict]) -> str:
    """
    Builds the user prompt for a batch of up to 5 sentences.
    Each sentence is numbered for unambiguous matching in the response.
    """
    parts = ["Classify each of the following disclosure sentences.\n"]
    for i, s in enumerate(sentences, 1):
        section = s.get("section", "not specified")
        parts.append(f"Sentence {i} (Section: {section}):")
        parts.append(f'"""{s["text"].strip()}"""\n')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSONL GENERATION
# ---------------------------------------------------------------------------

def generate_batch_jsonl(df: pd.DataFrame, output_path: str, run_id: str) -> int:
    """
    Creates a JSONL file for the OpenAI Batch API.
    Each line is one request containing up to BATCH_SIZE sentences.
    Returns the number of requests generated.
    """
    requests = []
    rows = df.to_dict("records")

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        custom_id = f"{run_id}_batch_{batch_start:06d}"

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(batch)},
                ],
            },
        }
        requests.append(request)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    return len(requests)


# ---------------------------------------------------------------------------
# BATCH API OPERATIONS
# ---------------------------------------------------------------------------

def upload_and_create_batch(client: OpenAI, jsonl_path: str,
                             description: str) -> dict:
    """Uploads a JSONL file and creates a batch job. Returns batch metadata."""
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )

    return {
        "batch_id":      batch.id,
        "input_file_id": file_obj.id,
        "status":        batch.status,
    }


def check_batch_status(client: OpenAI, batch_id: str) -> dict:
    """Returns current batch status and metadata."""
    batch = client.batches.retrieve(batch_id)
    return {
        "status":           batch.status,
        "total":            batch.request_counts.total if batch.request_counts else 0,
        "completed":        batch.request_counts.completed if batch.request_counts else 0,
        "failed":           batch.request_counts.failed if batch.request_counts else 0,
        "output_file_id":   batch.output_file_id,
        "error_file_id":    batch.error_file_id,
    }


def download_batch_results(client: OpenAI, output_file_id: str,
                            save_path: str) -> list[dict]:
    """Downloads and parses batch results. Caches to disk."""
    content = client.files.content(output_file_id)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content.read())

    results = []
    with open(save_path, "r") as f:
        for line in f:
            results.append(json.loads(line))

    return results


# ---------------------------------------------------------------------------
# RESULT PARSING
# ---------------------------------------------------------------------------

def parse_batch_results(results: list[dict], df: pd.DataFrame,
                         batch_size: int) -> dict:
    """
    Parses batch API results back into per-sentence labels.
    Returns dict mapping row index -> {label, confidence, reason}
    """
    labels = {}

    for result in results:
        custom_id = result["custom_id"]
        response  = result.get("response", {})

        if response.get("status_code") != 200:
            # Mark all sentences in this batch as failed
            batch_start = int(custom_id.split("_batch_")[1])
            for offset in range(min(batch_size, len(df) - batch_start)):
                labels[batch_start + offset] = {
                    "label": None, "confidence": None,
                    "reason": f"API error: {response.get('status_code')}",
                }
            continue

        # Extract the model's JSON response
        body = response.get("body", {})
        choices = body.get("choices", [])
        if not choices:
            continue

        raw_text = choices[0].get("message", {}).get("content", "")

        try:
            parsed = json.loads(raw_text)
            sentence_results = parsed.get("results", [])
        except json.JSONDecodeError:
            batch_start = int(custom_id.split("_batch_")[1])
            for offset in range(min(batch_size, len(df) - batch_start)):
                labels[batch_start + offset] = {
                    "label": None, "confidence": None,
                    "reason": "JSON parse error",
                }
            continue

        # Map sentence numbers back to DataFrame indices
        batch_start = int(custom_id.split("_batch_")[1])
        for sr in sentence_results:
            sentence_num = sr.get("sentence", 0)
            if sentence_num < 1:
                continue
            row_idx = batch_start + sentence_num - 1
            if row_idx < len(df):
                label_val = sr.get("label")
                if label_val not in [0, 1]:
                    label_val = None
                labels[row_idx] = {
                    "label":      label_val,
                    "confidence": sr.get("confidence"),
                    "reason":     sr.get("reason", ""),
                }

    return labels


# ---------------------------------------------------------------------------
# CONSENSUS AND EXPORT
# ---------------------------------------------------------------------------

def compute_consensus(labels_1: dict, labels_2: dict,
                       n_rows: int) -> pd.DataFrame:
    """
    Compares two independent labelling runs.
    Returns a DataFrame with consensus results for each sentence.
    """
    rows = []
    for idx in range(n_rows):
        r1 = labels_1.get(idx, {"label": None, "confidence": None, "reason": "missing"})
        r2 = labels_2.get(idx, {"label": None, "confidence": None, "reason": "missing"})

        l1 = r1["label"]
        l2 = r2["label"]

        if l1 is None or l2 is None:
            agreed = False
            final_label = None
            final_confidence = None
        else:
            agreed = (l1 == l2)
            final_label = l1 if agreed else None
            final_confidence = r1["confidence"] if agreed else None

        rows.append({
            "label":        final_label,
            "confidence":   final_confidence,
            "reason_1":     r1["reason"],
            "reason_2":     r2["reason"],
            "consensus":    agreed,
            "needs_review": not agreed,
        })

    return pd.DataFrame(rows)


def export_outputs(df: pd.DataFrame, consensus_df: pd.DataFrame,
                    output_dir: str):
    """
    Merges original data with consensus results and exports
    the three output CSVs in the format run_3 and run_4 expect.
    """
    merged = pd.concat([df.reset_index(drop=True), consensus_df], axis=1)
    os.makedirs(output_dir, exist_ok=True)

    # --- needs_review: consensus failed ---
    review_df = merged[merged["needs_review"] == True].copy()

    # --- labelled: consensus passed ---
    labelled_df = merged[merged["consensus"] == True].copy()

    # --- validation subset: 20% random sample from labelled ---
    np.random.seed(SEED)
    val_size = max(1, int(len(labelled_df) * VALIDATION_FRAC))
    val_idx  = np.random.choice(labelled_df.index, size=val_size, replace=False)
    val_df   = labelled_df.loc[val_idx].copy()

    val_cols = ["text"]
    if "section" in val_df.columns:
        val_cols.append("section")
    val_cols.extend(["label", "confidence", "reason_1"])

    val_export = val_df[val_cols].copy()
    val_export = val_export.rename(columns={
        "label":      "gpt4o_label",
        "confidence": "gpt4o_confidence",
        "reason_1":   "gpt4o_reason",
    })
    if "section" in val_export.columns:
        val_export = val_export.rename(columns={"section": "risk_category"})
    val_export["human_label"] = ""
    val_export["human_notes"] = ""

    # Save
    labelled_path = os.path.join(output_dir, "labelled.csv")
    val_path      = os.path.join(output_dir, "validation_subset.csv")
    review_path   = os.path.join(output_dir, "needs_review.csv")

    labelled_df.to_csv(labelled_path, index=False)
    val_export.to_csv(val_path,       index=False)

    if len(review_df):
        review_df.to_csv(review_path, index=False)
        print(f"  Needs review:       {len(review_df)} sentences  ->  {review_path}")
    else:
        print(f"  Needs review:       0 sentences (full consensus)")

    print(f"  Labelled:           {len(labelled_df)} sentences  ->  {labelled_path}")
    print(f"  Validation subset:  {len(val_export)} sentences  ->  {val_path}")

    return labelled_df, val_export, review_df


def print_summary(labelled_df: pd.DataFrame, review_df: pd.DataFrame):
    total     = len(labelled_df) + len(review_df)
    consensus = len(labelled_df)
    vague     = (labelled_df["label"] == 1).sum()
    specific  = (labelled_df["label"] == 0).sum()
    high_conf = (labelled_df["confidence"] == "high").sum()
    med_conf  = (labelled_df["confidence"] == "medium").sum()
    low_conf  = (labelled_df["confidence"] == "low").sum()

    print(f"\n{'='*55}")
    print(f"  Vagueness Labelling Summary")
    print(f"{'='*55}")
    print(f"  Total sentences:     {total}")
    print(f"  Consensus reached:   {consensus}  ({100*consensus/max(total,1):.1f}%)")
    print(f"  Flagged for review:  {len(review_df)}  ({100*len(review_df)/max(total,1):.1f}%)")
    print(f"\n  Label distribution (consensus sentences):")
    print(f"    VAGUE    (1):  {vague}  ({100*vague/max(consensus,1):.1f}%)")
    print(f"    SPECIFIC (0):  {specific}  ({100*specific/max(consensus,1):.1f}%)")
    print(f"\n  Confidence distribution:")
    print(f"    High:    {high_conf}")
    print(f"    Medium:  {med_conf}")
    print(f"    Low:     {low_conf}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------------------------

def save_state(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DDDS GPT-4o Vagueness Labelling Pipeline (Batch API)"
    )
    parser.add_argument("--input",      default=INPUT_CSV,
                        help="Path to training sentences CSV")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory for output CSVs")
    parser.add_argument("--state-file", default=STATE_FILE,
                        help="Path to batch state file")
    parser.add_argument("--use-cached", action="store_true",
                        help="Skip API calls, use existing results")
    parser.add_argument("--reprocess",  action="store_true",
                        help="Re-download and reprocess completed batches")
    args = parser.parse_args()

    print(f"\n[DDDS] Vagueness Labelling Pipeline (Batch API)")
    print(f"  Input:      {args.input}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Model:      {MODEL}")
    print(f"  Batch size: {BATCH_SIZE} sentences per request\n")

    # --- Load data ---
    print("[1/5] Loading data...")
    df = pd.read_csv(args.input)
    if "text" not in df.columns:
        raise ValueError(f"Input CSV must have a 'text' column. Found: {list(df.columns)}")
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df.reset_index(drop=True)
    print(f"  {len(df)} sentences loaded")

    n_requests = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  {n_requests} batched requests (x2 for consensus = {n_requests * 2} total)\n")

    # --- Check for cached results ---
    results_1_path = os.path.join(args.output_dir, "batch_results_run1.jsonl")
    results_2_path = os.path.join(args.output_dir, "batch_results_run2.jsonl")

    if args.use_cached:
        if os.path.exists(results_1_path) and os.path.exists(results_2_path):
            print("  Using cached batch results\n")
        else:
            print("  [!] --use-cached specified but result files not found:")
            print(f"      {results_1_path}")
            print(f"      {results_2_path}")
            return
    else:
        # --- Connect to OpenAI ---
        print("[2/5] Connecting to OpenAI...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
        client = OpenAI(api_key=api_key)
        print("  Connected\n")

        # --- Check existing state ---
        state = load_state(args.state_file)

        if state and not args.reprocess:
            print("  Found existing batch state — checking status...")
        else:
            # --- Generate JSONL files ---
            print("[3/5] Generating batch request files...")
            jsonl_1 = os.path.join(args.output_dir, "batch_requests_run1.jsonl")
            jsonl_2 = os.path.join(args.output_dir, "batch_requests_run2.jsonl")

            n1 = generate_batch_jsonl(df, jsonl_1, "run1")
            n2 = generate_batch_jsonl(df, jsonl_2, "run2")
            print(f"  Generated {n1} requests per file\n")

            # --- Upload and create batches ---
            print("[4/5] Uploading batches to OpenAI...")
            print("  Uploading run 1...", end=" ")
            b1 = upload_and_create_batch(
                client, jsonl_1,
                "DDDS vagueness labelling - consensus run 1"
            )
            print(f"batch_id={b1['batch_id']}")

            print("  Uploading run 2...", end=" ")
            b2 = upload_and_create_batch(
                client, jsonl_2,
                "DDDS vagueness labelling - consensus run 2"
            )
            print(f"batch_id={b2['batch_id']}")

            state = {
                "run1": b1,
                "run2": b2,
                "n_sentences": len(df),
                "n_requests":  n_requests,
            }
            save_state(state, args.state_file)
            print(f"\n  Batch IDs saved to {args.state_file}")

        # --- Poll for completion ---
        print(f"\n[5/5] Waiting for batches to complete...")
        print(f"  (This typically takes 1-3 hours. You can close the terminal")
        print(f"   and re-run this script later to check status.)\n")

        poll_start = time.time()
        prev_completed = 0

        while True:
            s1 = check_batch_status(client, state["run1"]["batch_id"])
            s2 = check_batch_status(client, state["run2"]["batch_id"])

            state["run1"]["status"] = s1["status"]
            state["run2"]["status"] = s2["status"]
            save_state(state, args.state_file)

            # Combined progress across both runs
            total_requests   = (s1["total"] or n_requests) + (s2["total"] or n_requests)
            total_completed  = s1["completed"] + s2["completed"]
            total_failed     = s1["failed"] + s2["failed"]
            pct              = 100 * total_completed / max(total_requests, 1)

            # Progress bar
            bar_width = 30
            filled    = int(bar_width * total_completed / max(total_requests, 1))
            bar       = "█" * filled + "░" * (bar_width - filled)

            # ETA estimate
            elapsed = time.time() - poll_start
            eta_str = ""
            if total_completed > prev_completed and total_completed > 0:
                rate = total_completed / max(elapsed, 1)
                remaining = (total_requests - total_completed) / rate
                if remaining < 60:
                    eta_str = f"  ETA: <1 min"
                elif remaining < 3600:
                    eta_str = f"  ETA: ~{int(remaining / 60)} min"
                else:
                    eta_str = f"  ETA: ~{remaining / 3600:.1f} hrs"

            print(f"  [{bar}] {pct:5.1f}%  "
                  f"({total_completed}/{total_requests} requests)"
                  f"{eta_str}")
            print(f"    Run 1: {s1['status']:<12} {s1['completed']}/{s1['total'] or '?'}   "
                  f"Run 2: {s2['status']:<12} {s2['completed']}/{s2['total'] or '?'}"
                  f"   Failed: {total_failed}")

            prev_completed = total_completed

            if s1["status"] == "completed" and s2["status"] == "completed":
                print(f"\n  Both batches complete! ({int(elapsed)}s elapsed)")

                # Download results
                print("  Downloading run 1 results...")
                download_batch_results(client, s1["output_file_id"], results_1_path)

                print("  Downloading run 2 results...")
                download_batch_results(client, s2["output_file_id"], results_2_path)

                print("  Results saved to cache\n")
                break

            elif s1["status"] in ("failed", "expired", "cancelled") or \
                 s2["status"] in ("failed", "expired", "cancelled"):
                print("\n  [!] A batch has failed/expired/been cancelled.")
                print("  Delete the state file and re-run to retry:")
                print(f"    del {args.state_file}")
                return

            else:
                print(f"  Next check in {POLL_INTERVAL}s...\n")
                time.sleep(POLL_INTERVAL)

    # --- Process results ---
    print("  Processing results...")

    with open(results_1_path, "r") as f:
        raw_results_1 = [json.loads(line) for line in f]
    with open(results_2_path, "r") as f:
        raw_results_2 = [json.loads(line) for line in f]

    labels_1 = parse_batch_results(raw_results_1, df, BATCH_SIZE)
    labels_2 = parse_batch_results(raw_results_2, df, BATCH_SIZE)

    print(f"  Run 1: {len(labels_1)} sentences labelled")
    print(f"  Run 2: {len(labels_2)} sentences labelled\n")

    # --- Consensus ---
    print("  Computing consensus...")
    consensus_df = compute_consensus(labels_1, labels_2, len(df))

    # --- Export ---
    print("  Exporting outputs...\n")
    labelled, validation, review = export_outputs(df, consensus_df, args.output_dir)
    print_summary(labelled, review)

    print("[Done] Vagueness labelling complete.")
    print(f"\nNext step: have all three team members independently fill in the")
    print(f"'human_label' column in validation_subset.csv, then run:")
    print(f"  python run_3_-_human_agreement_checker.py \\")
    print(f"      --annotator1 brad.csv --annotator2 connor.csv \\")
    print(f"      --annotator3 ebro.csv --classifier vagueness")


if __name__ == "__main__":
    main()