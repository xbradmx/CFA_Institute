"""
Quick test v2 — same 50 sentences, updated prompt with NOT_CLASSIFIABLE option.

Usage:
    python test_vague_prompt_v2.py
"""

import json
import os
import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
INPUT_CSV = "data/training_sentences/training_sentences.csv"
N_SAMPLES = 50
BATCH_SIZE = 5
MODEL = "gpt-4o"
SEED = 42
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert financial analyst specialising in SEC disclosure quality assessment.

Your task is to classify disclosure sentences from company filings (10-K, 10-Q) as either SPECIFIC or VAGUE, following the Bushee, Gow, and Taylor (2018) framework for deliberate linguistic vagueness.

DEFINITIONS
-----------
VAGUE (1): Language that is deliberately non-committal, evasive, or imprecise in a way that obscures information about the company's financial position, risks, or outlook. Vague language avoids binding the company to concrete statements.

SPECIFIC (0): Language that provides concrete, verifiable information. This includes specific figures, defined timelines, named counterparties, quantified risks, or clear causal explanations.

NOT CLASSIFIABLE (2): The sentence is not substantive disclosure content and should be excluded from training data. Use this label ONLY for:
- Navigational or cross-reference sentences ("see Note 8", "as described below", "refer to Part II")
- Section headings, table headers, or formatting artefacts
- Transitional sentences that simply introduce content elsewhere in the filing ("We have discussed our results using the measures described below")
- Fragments or incomplete sentences from document parsing

CRITICAL: Label 2 is NOT an escape for borderline cases. Any sentence that substantively discusses risks, operations, financial performance, strategy, or outlook MUST receive a 0 or 1, even if you find it difficult to classify. Reserve label 2 strictly for sentences that carry no analytical content whatsoever.

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

NOT CLASSIFIABLE: "You should read the following discussion and analysis of our financial condition and results of operations together with our consolidated financial statements."
Reason: Navigational sentence directing the reader, carries no analytical content.

NOT CLASSIFIABLE: "Quantitative and Qualitative Disclosures About Market Risk 41 Item 8."
Reason: Section heading and page number, not a substantive disclosure sentence.

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
    {"sentence": 3, "label": 2, "confidence": "high", "reason": "Navigational sentence referencing another section"}
  ]
}"""


def build_user_prompt(sentences):
    parts = ["Classify each of the following disclosure sentences.\n"]
    for i, s in enumerate(sentences, 1):
        section = s.get("section", "not specified")
        parts.append(f"Sentence {i} (Section: {section}):")
        parts.append(f'"""{s["text"].strip()}"""\n')
    return "\n".join(parts)


def main():
    # Load and sample — same seed as v1 so we get the same 50 sentences
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["text"])
    sample = df.sample(n=min(N_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)
    print(f"Sampled {len(sample)} sentences\n")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    all_results = []
    rows = sample.to_dict("records")

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"[Batch {batch_num}/{total_batches}] Sending {len(batch)} sentences...", end=" ")

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(batch)},
            ],
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        results = parsed.get("results", [])
        print(f"{len(results)} labels returned")

        for sr in results:
            idx = batch_start + sr["sentence"] - 1
            if idx < len(sample):
                all_results.append({
                    "idx":        idx,
                    "label":      sr["label"],
                    "confidence": sr["confidence"],
                    "reason":     sr["reason"],
                    "text":       sample.loc[idx, "text"],
                    "section":    sample.loc[idx, "section"],
                    "company":    sample.loc[idx, "company"] if "company" in sample.columns else "",
                })

    # Print results
    vague_count    = sum(1 for r in all_results if r["label"] == 1)
    specific_count = sum(1 for r in all_results if r["label"] == 0)
    skip_count     = sum(1 for r in all_results if r["label"] == 2)

    print(f"\n{'='*80}")
    print(f"  RESULTS: {specific_count} SPECIFIC  |  {vague_count} VAGUE  |  {skip_count} NOT CLASSIFIABLE  |  {len(all_results)} total")
    print(f"{'='*80}\n")

    for r in all_results:
        if r["label"] == 2:
            label_str = "SKIP    "
        elif r["label"] == 1:
            label_str = "VAGUE   "
        else:
            label_str = "SPECIFIC"

        conf = r["confidence"].upper()
        text_preview = r["text"][:120].replace("\n", " ")
        if len(r["text"]) > 120:
            text_preview += "..."

        print(f"  [{label_str}] [{conf:<6}]  {text_preview}")
        print(f"    Reason: {r['reason']}")
        print()


if __name__ == "__main__":
    main()