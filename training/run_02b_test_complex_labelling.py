"""
Quick live test — 50 sentences, Simple vs Complex labelling.
Runs instantly (NO async Batch API). Includes NOT_CLASSIFIABLE and refined prompt.

Usage:
    python test_complex_prompt.py
"""

import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
INPUT_CSV = "data/training_sentences/training_sentences.csv"
N_SAMPLES = 50
CHUNK_SIZE = 5  # Groups 5 sentences per instant API call
MODEL = "gpt-4o"
SEED = 42
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert financial analyst specialising in SEC disclosure quality assessment.

Your task is to classify disclosure passages from company filings (10-K, 10-Q) as SIMPLE, COMPLEX, or NOT CLASSIFIABLE, following the Bushee, Gow, and Taylor (2018) framework for linguistic complexity in firm disclosures.

DEFINITIONS
-----------
COMPLEX (1): Language that requires significant domain expertise, technical knowledge, or cognitive effort to process. Complexity arises from genuine informational density, technical terminology, convoluted syntax, or intricate cross-references. Complexity is not inherently negative - it reflects the legitimate difficulty of the underlying subject matter.

SIMPLE (0): Language that can be understood by a reasonably informed reader without specialist technical knowledge. Simple language communicates clearly and directly, regardless of whether the content is positive or negative.

NOT CLASSIFIABLE (2): The sentence is not substantive disclosure content and should be excluded from training data. Use this label ONLY for:
- Navigational or cross-reference sentences ("see Note 8", "as described below", "refer to Part II")
- Section headings, table headers, or formatting artefacts
- Transitional sentences that simply introduce content elsewhere in the filing
- Fragments or incomplete sentences from document parsing

CRITICAL: Label 2 is NOT an escape for borderline cases. Any sentence that substantively discusses risks, operations, financial performance, strategy, or outlook MUST receive a 0 or 1, even if you find it difficult to classify. Reserve label 2 strictly for sentences that carry no analytical content whatsoever.

COMPLEXITY INDICATORS - language is likely COMPLEX if it:
- Uses domain-specific accounting, legal, or engineering terminology that requires specialist knowledge to interpret (e.g., Black-Scholes, valuation allowances, deferred tax assets).
- Contains highly convoluted or deeply nested conditional structures that actively impede reading comprehension: "if X occurs, then Y applies, unless Z, in which case W..."
- References multiple interacting financial instruments, regulations, or standards simultaneously.
- Involves actuarial, derivative, or structured finance concepts requiring technical expertise.

SIMPLICITY INDICATORS - language is likely SIMPLE if it:
- Uses plain language accessible to a general business reader.
- Describes straightforward operational or commercial facts.
- Uses common financial terms (revenue, profit, debt, margins) without technical elaboration.
- Narrates events, decisions, or risks in a clear causal sequence without technical jargon.

IMPORTANT DISTINCTION: OPERATIONAL VS. LINGUISTIC COMPLEXITY
------------------------------------------------------------
Do not confuse *operational* complexity with *linguistic* complexity. Describing a difficult business challenge, a multi-step process, or a hypothetical risk in plain, accessible English is STILL SIMPLE. Only classify as COMPLEX if the vocabulary or sentence structure genuinely requires a finance, legal, or accounting specialist to decipher.

IMPORTANT DISTINCTION: VAGUENESS VS. COMPLEXITY
-----------------------------------------------
Do not confuse complexity with vagueness. Simple language can be vague ("results may vary") and complex language can be specific ("the fair value of the swap, determined using a Level 2 DCF model discounting at 5.23%, was $14.7M"). Classify solely on cognitive processing difficulty, not evasiveness.

EXAMPLES
--------
COMPLEX: "The Company's net periodic pension cost includes service cost, interest cost, expected return on plan assets, amortisation of prior service cost or credit, and amortisation of actuarial gains or losses recognised in accumulated other comprehensive income under the corridor approach, with the corridor defined as 10% of the greater of the projected benefit obligation or market-related value of plan assets."
Reason: Requires actuarial expertise, references multiple interacting components, uses specialist accounting terminology throughout.

SIMPLE: "If we are unable to continue to develop self-service support resources that are easy to use, our users may experience decreased satisfaction, which could negatively impact our retention rates."
Reason: While this describes a multi-step hypothetical risk (operational complexity), the language is entirely plain and accessible to a general reader (linguistic simplicity).

NOT CLASSIFIABLE: "You should read the following discussion and analysis of our financial condition and results of operations together with our consolidated financial statements."
Reason: Navigational sentence directing the reader, carries no analytical content.

BORDERLINE GUIDANCE
-------------------
If a passage uses some technical terms but the overall meaning is accessible to a reasonably informed investor, classify as SIMPLE. Only classify as COMPLEX where the technical density genuinely creates a significant processing barrier for a non-specialist reader.

OUTPUT FORMAT
-------------
You will receive multiple numbered sentences. Classify each one independently.

Respond ONLY with a valid JSON object containing a "results" array. Each element must have the sentence number, label, confidence, and reason. Example:

{
  "results": [
    {"sentence": 1, "label": 0, "confidence": "high", "reason": "Straightforward explanation using common financial terms"},
    {"sentence": 2, "label": 1, "confidence": "high", "reason": "Requires actuarial expertise and references multiple interacting components"},
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
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["text"])
    sample = df.sample(n=min(N_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)
    print(f"Sampled {len(sample)} sentences for instant testing\n")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    all_results = []
    rows = sample.to_dict("records")

    for chunk_start in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[chunk_start : chunk_start + CHUNK_SIZE]
        chunk_num = chunk_start // CHUNK_SIZE + 1
        total_chunks = (len(rows) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"[Group {chunk_num}/{total_chunks}] Processing {len(chunk)} sentences instantly...", end=" ", flush=True)

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(chunk)},
            ],
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        results = parsed.get("results", [])
        print(f"Done! ({len(results)} labels returned)")

        for sr in results:
            idx = chunk_start + sr["sentence"] - 1
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

    complex_count  = sum(1 for r in all_results if r["label"] == 1)
    simple_count   = sum(1 for r in all_results if r["label"] == 0)
    skip_count     = sum(1 for r in all_results if r["label"] == 2)

    print(f"\n{'='*90}")
    print(f"  RESULTS: {simple_count} SIMPLE  |  {complex_count} COMPLEX  |  {skip_count} NOT CLASSIFIABLE  |  {len(all_results)} total")
    print(f"{'='*90}\n")

    for r in all_results:
        if r["label"] == 2:
            label_str = "SKIP    "
        elif r["label"] == 1:
            label_str = "COMPLEX "
        else:
            label_str = "SIMPLE  "

        conf = r["confidence"].upper()
        text_preview = r["text"][:120].replace("\n", " ")
        if len(r["text"]) > 120:
            text_preview += "..."

        print(f"  [{label_str}] [{conf:<6}]  {text_preview}")
        print(f"    Reason: {r['reason']}")
        print()


if __name__ == "__main__":
    main()