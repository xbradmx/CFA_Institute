"""
DDDS - GPT-4o Complexity Labelling Pipeline
============================================
Task:   Label disclosure passages as SIMPLE (0) or COMPLEX (1)
Model:  GPT-4o (consensus labelling with confidence scoring)

What this script does
---------------------
1. Reads an input CSV of unlabelled disclosure passages
2. Sends each passage to GPT-4o with a detailed prompt grounded in
   the Bushee et al. (2018) complexity framework
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
    python label_complexity.py --input data/passages.csv
    python label_complexity.py --input data/passages.csv --output-dir data/labelled/complexity
"""

import argparse
import os
import time
import json
import pandas as pd
import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL           = "gpt-4o"
INPUT_CSV       = "data/passages.csv"
OUTPUT_DIR      = "data/labelled/complexity"
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

Your task is to classify disclosure passages from company filings (10-K, 10-Q) as either SIMPLE or COMPLEX, following the Bushee, Gow, and Taylor (2018) framework for linguistic complexity in firm disclosures.

DEFINITIONS
-----------
COMPLEX (1): Language that requires significant domain expertise, technical knowledge, or cognitive effort to process. Complexity arises from genuine informational density, technical terminology, layered conditional structures, or intricate cross-references. Complexity is not inherently negative - it reflects the legitimate difficulty of the underlying subject matter.

SIMPLE (0): Language that can be understood by a reasonably informed reader without specialist technical knowledge. Simple language communicates clearly and directly, regardless of whether the content is positive or negative.

COMPLEXITY INDICATORS - language is likely COMPLEX if it:
- Uses domain-specific accounting, legal, or engineering terminology that requires specialist knowledge to interpret
- Contains layered conditional structures: "if X occurs, then Y applies, unless Z, in which case W"
- References multiple interacting financial instruments, regulations, or standards simultaneously
- Requires cross-referencing other parts of the filing to understand the statement
- Uses nested clauses that substantially increase sentence processing load
- Involves actuarial, derivative, or structured finance concepts requiring technical expertise
- Contains dense numerical relationships requiring calculation to interpret (ratios, sensitivities, stress tests)

SIMPLICITY INDICATORS - language is likely SIMPLE if it:
- Uses plain language accessible to a general business reader
- Describes straightforward operational or commercial facts
- Presents information in a single direct clause without significant conditionals
- Uses common financial terms (revenue, profit, debt) without technical elaboration
- Narrates events or decisions in chronological or causal sequence without technical nesting

IMPORTANT DISTINCTION
---------------------
Do not confuse complexity with vagueness. Simple language can be vague ("results may vary") and complex language can be specific ("the fair value of the interest rate swap, determined using a Level 2 DCF model discounting expected SOFR-linked cash flows at 5.23%, was $14.7 million as of 31 December 2024"). Classify solely on cognitive processing difficulty, not on evasiveness or specificity.

EXAMPLES
--------
COMPLEX: "The Company's net periodic pension cost includes service cost, interest cost, expected return on plan assets, amortisation of prior service cost or credit, and amortisation of actuarial gains or losses recognised in accumulated other comprehensive income under the corridor approach, with the corridor defined as 10% of the greater of the projected benefit obligation or market-related value of plan assets."
Reason: Requires actuarial expertise, references multiple interacting components, uses specialist accounting terminology throughout.

COMPLEX: "We have entered into cross-currency interest rate swap agreements to hedge our exposure to variability in expected future cash flows attributable to changes in foreign currency exchange rates and interest rates on our CHF 300 million 1.5% senior notes due 2031, effectively converting the fixed-rate CHF obligation into a floating-rate USD obligation."
Reason: Requires understanding of derivative instruments, cross-currency mechanics, and hedging accounting.

SIMPLE: "Our revenue decreased by $23 million compared to the prior year, primarily due to lower sales volumes in our North American segment."
Reason: Straightforward causal explanation using common financial terms, no specialist knowledge required.

SIMPLE: "We are subject to various environmental regulations in the jurisdictions where we operate. Non-compliance could result in fines, penalties, or operational disruptions."
Reason: Plain language description of a standard regulatory risk, accessible to any business reader.

BORDERLINE GUIDANCE
-------------------
If a passage uses some technical terms but the overall meaning is accessible to a reasonably informed investor, classify as SIMPLE. Only classify as COMPLEX where the technical density genuinely creates a significant processing barrier for a non-specialist reader.

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
                ],
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
            "label":        None,
            "confidence":   None,
            "reason_1":     call_1.get("reason") if call_1 else "API failure",
            "reason_2":     call_2.get("reason") if call_2 else "API failure",
            "consensus":    False,
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

        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), results_df], axis=1)


def export_outputs(df: pd.DataFrame, output_dir: str):
    """
    Splits results into three output CSVs and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)

    review_df   = df[df["needs_review"] == True].copy()
    labelled_df = df[df["consensus"] == True].copy()

    np.random.seed(SEED)
    val_size = max(1, int(len(labelled_df) * VALIDATION_FRAC))
    val_idx  = np.random.choice(labelled_df.index, size=val_size, replace=False)
    val_df   = labelled_df.loc[val_idx].copy()

    val_export = val_df[["text", "risk_category", "label", "confidence", "reason_1"]].copy()
    val_export = val_export.rename(columns={
        "label":      "gpt4o_label",
        "confidence": "gpt4o_confidence",
        "reason_1":   "gpt4o_reason",
    })
    val_export["human_label"] = ""
    val_export["human_notes"] = ""

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
    total     = len(labelled_df) + len(review_df)
    consensus = len(labelled_df)
    complex_  = (labelled_df["label"] == 1).sum()
    simple    = (labelled_df["label"] == 0).sum()
    high_conf = (labelled_df["confidence"] == "high").sum()
    med_conf  = (labelled_df["confidence"] == "medium").sum()
    low_conf  = (labelled_df["confidence"] == "low").sum()

    print(f"\n{'='*55}")
    print(f"  Labelling Summary")
    print(f"{'='*55}")
    print(f"  Total passages:      {total}")
    print(f"  Consensus reached:   {consensus}  ({100*consensus/total:.1f}%)")
    print(f"  Flagged for review:  {len(review_df)}  ({100*len(review_df)/total:.1f}%)")
    print(f"\n  Label distribution (consensus passages):")
    print(f"    COMPLEX (1):  {complex_}  ({100*complex_/max(consensus,1):.1f}%)")
    print(f"    SIMPLE  (0):  {simple}  ({100*simple/max(consensus,1):.1f}%)")
    print(f"\n  Confidence distribution:")
    print(f"    High:    {high_conf}")
    print(f"    Medium:  {med_conf}")
    print(f"    Low:     {low_conf}")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(description="DDDS GPT-4o Complexity Labelling Pipeline")
    parser.add_argument("--input",      default=INPUT_CSV, help="Path to input CSV of passages")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for output CSVs")
    args = parser.parse_args()

    print(f"\n[DDDS] Complexity Labelling Pipeline")
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
    print("[Done] Complexity labelling complete.")
    print(f"\nNext step: have all three team members independently fill in the")
    print(f"'human_label' column in validation_subset.csv, then run check_agreement.py")


if __name__ == "__main__":
    main()
