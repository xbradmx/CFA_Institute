"""
DDDS - GPT-4o Vagueness Labelling Pipeline
===========================================
Task:   Label disclosure passages as SPECIFIC (0) or VAGUE (1)
Model:  GPT-4o (consensus labelling with confidence scoring)

What this script does
---------------------
1. Reads an input CSV of unlabelled disclosure passages
2. Sends each passage to GPT-4o with a detailed prompt grounded in
   the Bushee et al. (2018) vagueness framework
3. Runs each passage twice and only accepts a label if both calls agree
   (consensus labelling) - disagreements are flagged for human review
4. Exports three CSVs:
     - labelled.csv         : all passages with accepted GPT-4o labels
     - validation_subset.csv: 20% random sample with GPT-4o labels pre-filled
                              for your team to independently review
     - needs_review.csv     : passages where GPT-4o calls disagreed

Input CSV must have:
    text          : str  - the disclosure passage
    risk_category : str  - one of the six predefined categories (optional but recommended)

Usage
-----
    python label_vagueness.py --input data/passages.csv
    python label_vagueness.py --input data/passages.csv --output-dir data/labelled/vagueness
"""

import argparse
import os
import time
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL           = "gpt-4o"
INPUT_CSV       = "data/passages.csv"
OUTPUT_DIR      = "data/labelled/vagueness"
VALIDATION_FRAC = 0.20       # fraction exported for human validation
TEMPERATURE     = 0.0        # deterministic outputs
MAX_RETRIES     = 3          # retries on API failure
RETRY_DELAY     = 5          # seconds between retries
SEED            = 42
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert financial analyst specialising in SEC disclosure quality assessment.

Your task is to classify disclosure passages from company filings (10-K, 10-Q) as either SPECIFIC or VAGUE, following the Bushee, Gow, and Taylor (2018) framework for deliberate linguistic vagueness.

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
If a passage mixes specific and vague elements, classify based on the dominant character. If a passage contains one specific figure embedded in otherwise entirely evasive language, lean VAGUE. If a passage is technically hedged but provides substantive specific context, lean SPECIFIC.

OUTPUT FORMAT
-------------
Respond only with a valid JSON object. No preamble, no explanation outside the JSON.
{
  "label": 0 or 1,
  "confidence": "high" or "medium" or "low",
  "reason": "one sentence explaining the classification"
}"""

USER_PROMPT_TEMPLATE = """Classify the following disclosure passage.

Risk category: {risk_category}

Passage:
\"\"\"{text}\"\"\"
"""
# ---------------------------------------------------------------------------


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )
    return OpenAI(api_key=api_key)


def call_gpt4o(client: OpenAI, text: str, risk_category: str) -> dict | None:
    """
    Single API call to GPT-4o. Returns parsed JSON dict or None on failure.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        text=text.strip(),
        risk_category=risk_category if risk_category else "not specified"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)

            # Validate expected fields
            if "label" not in result or result["label"] not in [0, 1]:
                raise ValueError(f"Unexpected label value: {result.get('label')}")

            return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [!] API call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [!] All retries failed for passage. Marking for review.")
                return None


def label_passage(client: OpenAI, text: str, risk_category: str) -> dict:
    """
    Runs two independent GPT-4o calls on the same passage.
    Returns consensus result or flags disagreement for human review.
    """
    call_1 = call_gpt4o(client, text, risk_category)
    call_2 = call_gpt4o(client, text, risk_category)

    # If either call failed entirely
    if call_1 is None or call_2 is None:
        return {
            "label":      None,
            "confidence": None,
            "reason_1":   call_1.get("reason") if call_1 else "API failure",
            "reason_2":   call_2.get("reason") if call_2 else "API failure",
            "consensus":  False,
            "needs_review": True,
        }

    agreed = call_1["label"] == call_2["label"]

    return {
        "label":        call_1["label"] if agreed else None,
        "confidence":   call_1["confidence"] if agreed else None,
        "reason_1":     call_1["reason"],
        "reason_2":     call_2["reason"],
        "consensus":    agreed,
        "needs_review": not agreed,
    }


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError(
            f"Input CSV must have a 'text' column. Found: {list(df.columns)}"
        )

    # Add risk_category column if missing
    if "risk_category" not in df.columns:
        print("  [!] No 'risk_category' column found - will use 'not specified' for all rows")
        df["risk_category"] = "not specified"

    before = len(df)
    df = df.dropna(subset=["text"])
    dropped = before - len(df)
    if dropped:
        print(f"  [!] Dropped {dropped} rows with missing text")

    df["text"] = df["text"].astype(str).str.strip()
    df = df.reset_index(drop=True)
    return df


def run_labelling(df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """
    Iterates over all passages, calls GPT-4o twice per passage,
    and appends results as new columns.
    """
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        print(f"  [{idx+1}/{total}] Labelling passage...", end=" ")

        result = label_passage(client, row["text"], row["risk_category"])
        results.append(result)

        status = "✓ consensus" if result["consensus"] else "✗ disagreement - flagged"
        print(status)

        # Small delay to respect rate limits
        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), results_df], axis=1)


def export_outputs(df: pd.DataFrame, output_dir: str):
    """
    Splits results into three output CSVs and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- needs_review: consensus failed ---
    review_df = df[df["needs_review"] == True].copy()

    # --- labelled: consensus passed ---
    labelled_df = df[df["consensus"] == True].copy()

    # --- validation subset: 20% random sample from labelled ---
    np.random.seed(SEED)
    val_size = max(1, int(len(labelled_df) * VALIDATION_FRAC))
    val_idx  = np.random.choice(labelled_df.index, size=val_size, replace=False)
    val_df   = labelled_df.loc[val_idx].copy()

    # Validation CSV columns: text, risk_category, gpt4o_label, confidence, reason_1
    # Human annotators see the GPT-4o label and reason so they can agree or override
    val_export = val_df[["text", "risk_category", "label", "confidence", "reason_1"]].copy()
    val_export = val_export.rename(columns={
        "label":      "gpt4o_label",
        "confidence": "gpt4o_confidence",
        "reason_1":   "gpt4o_reason",
    })
    val_export["human_label"]   = ""   # annotator fills this in
    val_export["human_notes"]   = ""   # optional annotator comments

    # Save
    labelled_path = os.path.join(output_dir, "labelled.csv")
    val_path      = os.path.join(output_dir, "validation_subset.csv")
    review_path   = os.path.join(output_dir, "needs_review.csv")

    labelled_df.to_csv(labelled_path, index=False)
    val_export.to_csv(val_path,       index=False)

    if len(review_df):
        review_df.to_csv(review_path, index=False)
        print(f"  Needs review:       {len(review_df)} passages  ->  {review_path}")
    else:
        print(f"  Needs review:       0 passages (no disagreements)")

    print(f"  Labelled:           {len(labelled_df)} passages  ->  {labelled_path}")
    print(f"  Validation subset:  {len(val_export)} passages  ->  {val_path}")

    return labelled_df, val_export, review_df


def print_summary(labelled_df: pd.DataFrame, review_df: pd.DataFrame):
    total      = len(labelled_df) + len(review_df)
    consensus  = len(labelled_df)
    vague      = (labelled_df["label"] == 1).sum()
    specific   = (labelled_df["label"] == 0).sum()
    high_conf  = (labelled_df["confidence"] == "high").sum()
    med_conf   = (labelled_df["confidence"] == "medium").sum()
    low_conf   = (labelled_df["confidence"] == "low").sum()

    print(f"\n{'='*55}")
    print(f"  Labelling Summary")
    print(f"{'='*55}")
    print(f"  Total passages:      {total}")
    print(f"  Consensus reached:   {consensus}  ({100*consensus/total:.1f}%)")
    print(f"  Flagged for review:  {len(review_df)}  ({100*len(review_df)/total:.1f}%)")
    print(f"\n  Label distribution (consensus passages):")
    print(f"    VAGUE    (1):  {vague}  ({100*vague/max(consensus,1):.1f}%)")
    print(f"    SPECIFIC (0):  {specific}  ({100*specific/max(consensus,1):.1f}%)")
    print(f"\n  Confidence distribution:")
    print(f"    High:    {high_conf}")
    print(f"    Medium:  {med_conf}")
    print(f"    Low:     {low_conf}")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(description="DDDS GPT-4o Vagueness Labelling Pipeline")
    parser.add_argument("--input",      default=INPUT_CSV, help="Path to input CSV of passages")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for output CSVs")
    args = parser.parse_args()

    print(f"\n[DDDS] Vagueness Labelling Pipeline")
    print(f"  Input:      {args.input}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Model:      {MODEL}\n")

    print("[1/4] Loading data...")
    df = load_data(args.input)
    print(f"  {len(df)} passages loaded\n")

    print("[2/4] Connecting to OpenAI...")
    client = build_client()
    print("  Connected\n")

    print("[3/4] Labelling passages (2 API calls per passage)...")
    labelled_df = run_labelling(df, client)

    print("\n[4/4] Exporting outputs...")
    labelled, validation, review = export_outputs(labelled_df, args.output_dir)

    print_summary(labelled, review)
    print("[Done] Vagueness labelling complete.")
    print(f"\nNext step: have all three team members independently fill in the")
    print(f"'human_label' column in validation_subset.csv, then run check_agreement.py")


if __name__ == "__main__":
    main()
