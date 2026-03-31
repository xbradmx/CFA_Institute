"""
DDDS - Graph-RAG Analysis Pipeline
====================================
Implements the two-tier LLM analysis layer on top of the Neo4j graph.

Architecture
------------
Tier 1 - GPT-4o-mini:
    Screens each company's RiskFactor nodes for material changes across
    quarters. Fast and cheap. Flags items that warrant deep analysis.

Tier 2 - Claude Opus:
    Receives flagged items and performs deep analysis:
      - Temporal comparison (quarter-to-quarter language changes)
      - Cross-document contradiction detection (10-K vs earnings call)
      - Peer-relative analysis (divergence from sector norms)

Outputs per flagged company:
    - Structured finding dict (JSON)
    - Saved to data/findings/<identifier>_findings.json

Usage
-----
    python graph_rag.py --company AAPL
    python graph_rag.py --all
    python graph_rag.py --all --output-dir data/findings
"""

import argparse
import os
import json
import time
import tiktoken
from neo4j import GraphDatabase
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
load_dotenv()


NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

SCREENING_MODEL   = "gpt-4o-mini"
DEEP_ANALYSIS_MODEL = "claude-opus-4-6"

SCREENING_THRESHOLD = 0.7    # GPT-4o-mini confidence above which item is escalated
MAX_TOKENS_CONTEXT  = 180000  # Claude Opus context window safety limit
OUTPUT_DIR          = "data/findings"
RETRY_DELAY         = 5
MAX_RETRIES         = 3
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NEO4J QUERIES
# ---------------------------------------------------------------------------

class GraphQuerier:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_all_companies(self) -> list[dict]:
        query = """
        MATCH (c:Company)
        RETURN c.identifier AS identifier, c.name AS name, c.sector AS sector
        ORDER BY c.identifier
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query)]

    def get_company_filings(self, identifier: str) -> list[dict]:
        """Returns all filings for a company ordered by period."""
        query = """
        MATCH (c:Company {identifier: $identifier})-[:HAS_FILING]->(f:Filing)
        RETURN f.filing_id AS filing_id,
               f.filename  AS filename,
               f.filing_type AS filing_type,
               f.period    AS period
        ORDER BY f.period ASC
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, identifier=identifier)]

    def get_filing_risk_factors(self, filing_id: str) -> list[dict]:
        """Returns all RiskFactor nodes for a filing."""
        query = """
        MATCH (f:Filing {filing_id: $filing_id})-[:HAS_RISK_FACTOR]->(r:RiskFactor)
        RETURN r.rf_id        AS rf_id,
               r.section      AS section,
               r.risk_category AS risk_category,
               r.text         AS text,
               r.vague_label  AS vague_label,
               r.vague_prob   AS vague_prob,
               r.complex_label AS complex_label,
               r.complex_prob  AS complex_prob
        ORDER BY r.risk_category
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, filing_id=filing_id)]

    def get_temporal_pairs(self, identifier: str) -> list[tuple[dict, dict]]:
        """
        Returns consecutive filing pairs (prev, curr) for a company
        via NEXT_PERIOD edges.
        """
        query = """
        MATCH (c:Company {identifier: $identifier})-[:HAS_FILING]->(f1:Filing)
              -[:NEXT_PERIOD]->(f2:Filing)
        RETURN f1.filing_id AS prev_id, f1.period AS prev_period,
               f1.filing_type AS prev_type,
               f2.filing_id AS curr_id, f2.period AS curr_period,
               f2.filing_type AS curr_type
        ORDER BY f2.period ASC
        """
        with self.driver.session() as session:
            return [(dict(r)["prev_id"], dict(r)["curr_id"], dict(r))
                    for r in session.run(query, identifier=identifier)]

    def get_peer_risk_factors(self, identifier: str,
                               risk_category: str,
                               period: str) -> list[dict]:
        """
        Returns RiskFactor nodes from peer companies in the same
        risk_category and period for benchmarking.
        """
        query = """
        MATCH (c:Company {identifier: $identifier})-[:PEER_OF]->(peer:Company)
              -[:HAS_FILING]->(f:Filing {period: $period})
              -[:HAS_RISK_FACTOR]->(r:RiskFactor {risk_category: $risk_category})
        RETURN peer.identifier AS peer_id,
               peer.name       AS peer_name,
               r.text          AS text,
               r.vague_prob    AS vague_prob,
               r.complex_prob  AS complex_prob
        LIMIT 5
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(
                query,
                identifier=identifier,
                risk_category=risk_category,
                period=period
            )]

    def get_same_quarter_transcript(self, identifier: str,
                                     period: str) -> list[dict]:
        """
        Returns transcript RiskFactor nodes for the same company and period,
        used for cross-document contradiction detection.
        """
        query = """
        MATCH (c:Company {identifier: $identifier})
              -[:HAS_FILING]->(f:Filing {period: $period, filing_type: 'earnings_transcript'})
              -[:HAS_RISK_FACTOR]->(r:RiskFactor)
        RETURN r.text AS text, r.risk_category AS risk_category
        ORDER BY r.risk_category
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(
                query, identifier=identifier, period=period
            )]


# ---------------------------------------------------------------------------
# TIER 1 - GPT-4o-mini SCREENING
# ---------------------------------------------------------------------------

SCREENING_SYSTEM_PROMPT = """You are a financial analyst screening disclosure passages for material changes.

You will be given two versions of a company's risk factor disclosure:
- PREVIOUS PERIOD passage
- CURRENT PERIOD passage

Your task is to assess whether there is a material change between the two passages that warrants deep analysis.

A change is MATERIAL if it involves:
- New risks added that were not present before
- Specific metrics removed and replaced with vague language
- Significant intensification of language around an existing risk
- Removal of previously disclosed specific commitments or figures
- Introduction of going concern or solvency language
- New legal, regulatory, or environmental exposures

A change is NOT material if it involves:
- Minor rewording with no substantive difference
- Updated figures consistent with normal business progression
- Routine boilerplate language present in both periods

Respond only with a valid JSON object:
{
  "material_change": true or false,
  "confidence": 0.0 to 1.0,
  "change_type": "new_risk" | "language_intensification" | "metric_removal" | "solvency_language" | "no_change" | "other",
  "reason": "one sentence describing the change or lack of change"
}"""


def screen_temporal_change(client: OpenAI,
                            prev_text: str,
                            curr_text: str,
                            risk_category: str,
                            company: str) -> dict | None:
    """
    Tier 1: GPT-4o-mini screening of a single temporal pair.
    Returns screening result dict or None on failure.
    """
    user_prompt = f"""Company: {company}
Risk Category: {risk_category}

PREVIOUS PERIOD:
\"\"\"{prev_text[:2000]}\"\"\"

CURRENT PERIOD:
\"\"\"{curr_text[:2000]}\"\"\"
"""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=SCREENING_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SCREENING_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"      [!] Screening failed: {e}")
                return None


# ---------------------------------------------------------------------------
# TIER 2 - CLAUDE OPUS DEEP ANALYSIS
# ---------------------------------------------------------------------------

DEEP_ANALYSIS_SYSTEM_PROMPT = """You are a senior financial analyst conducting deep disclosure quality analysis for investment research.

You will receive a structured context containing:
1. TEMPORAL CONTEXT: Risk factor passages from consecutive quarters for a company
2. PEER CONTEXT: How peers in the same sector are disclosing the same risk category
3. TRANSCRIPT CONTEXT: What management said on the earnings call for the same period

Your task is to produce a structured deep analysis covering:

A) TEMPORAL ANALYSIS
   - What specifically changed between periods (language, figures, scope)
   - Whether changes represent deterioration, improvement, or obfuscation
   - Whether specific metrics were removed or replaced with vague language

B) CROSS-DOCUMENT CONTRADICTION
   - Whether the 10-K risk language is consistent with earnings call language
   - Flag any instances where management downplayed in the call what was disclosed in the filing

C) PEER-RELATIVE ASSESSMENT
   - Whether this company's disclosure is more or less specific than peers
   - Whether this company omits risks that peers uniformly disclose
   - Whether vagueness scores deviate significantly from peer averages

D) OVERALL SIGNAL
   - Summarise whether this represents a meaningful disclosure degradation signal
   - Rate signal strength: HIGH / MEDIUM / LOW
   - State what an analyst should investigate further

Be specific. Reference exact language from the passages. Do not produce generic commentary.

Respond in the following JSON structure:
{
  "temporal_analysis": {
    "changes_detected": [...],
    "deterioration_assessment": "...",
    "metric_removal": true or false,
    "metric_removal_detail": "..."
  },
  "contradiction_analysis": {
    "contradiction_detected": true or false,
    "contradiction_detail": "...",
    "supporting_evidence": ["filing quote", "transcript quote"]
  },
  "peer_analysis": {
    "peer_comparison": "...",
    "omitted_risks": [...],
    "vagueness_deviation": "above_average" | "average" | "below_average"
  },
  "overall_signal": {
    "signal_strength": "HIGH" | "MEDIUM" | "LOW",
    "summary": "...",
    "analyst_actions": [...]
  }
}"""


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Counts tokens using tiktoken to avoid context overflow."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def build_deep_analysis_context(company: dict,
                                 temporal_pairs: list,
                                 graph: GraphQuerier,
                                 screened_flags: list[dict]) -> str:
    """
    Assembles the full context payload for Claude Opus.
    Respects token limits by truncating peer/transcript context if needed.
    """
    sections = []
    sections.append(f"COMPANY: {company['name']} ({company['identifier']})")
    sections.append(f"SECTOR: {company['sector']}\n")

    # Temporal context - only flagged categories
    sections.append("=" * 60)
    sections.append("TEMPORAL CONTEXT - FLAGGED CHANGES")
    sections.append("=" * 60)

    for flag in screened_flags:
        sections.append(f"\nRisk Category: {flag['risk_category']}")
        sections.append(f"Period: {flag['prev_period']} → {flag['curr_period']}")
        sections.append(f"Change Type: {flag['change_type']}")
        sections.append(f"Screening Reason: {flag['screening_reason']}\n")
        sections.append(f"PREVIOUS ({flag['prev_period']}):")
        sections.append(flag['prev_text'][:1500])
        sections.append(f"\nCURRENT ({flag['curr_period']}):")
        sections.append(flag['curr_text'][:1500])
        sections.append("-" * 40)

    # Peer context
    sections.append("\n" + "=" * 60)
    sections.append("PEER CONTEXT")
    sections.append("=" * 60)

    for flag in screened_flags[:3]:  # limit to top 3 flagged categories
        peers = graph.get_peer_risk_factors(
            company["identifier"],
            flag["risk_category"],
            flag["curr_period"]
        )
        if peers:
            sections.append(f"\nPeer disclosures for '{flag['risk_category']}' "
                           f"in {flag['curr_period']}:")
            for p in peers[:3]:
                sections.append(f"\n  [{p['peer_id']}] {p['text'][:500]}")

    # Transcript context
    sections.append("\n" + "=" * 60)
    sections.append("EARNINGS CALL TRANSCRIPT CONTEXT")
    sections.append("=" * 60)

    periods_checked = set()
    for flag in screened_flags:
        if flag["curr_period"] not in periods_checked:
            transcript = graph.get_same_quarter_transcript(
                company["identifier"], flag["curr_period"]
            )
            if transcript:
                sections.append(f"\nTranscript passages for {flag['curr_period']}:")
                for t in transcript[:5]:
                    sections.append(f"\n  [{t['risk_category']}] {t['text'][:500]}")
            periods_checked.add(flag["curr_period"])

    context = "\n".join(sections)

    # Token safety check
    token_count = count_tokens(context)
    if token_count > MAX_TOKENS_CONTEXT:
        print(f"      [!] Context too large ({token_count} tokens), truncating...")
        # Truncate to safe limit by cutting peer and transcript sections
        context = "\n".join(sections[:len(sections)//2])

    return context


def run_deep_analysis(client: anthropic.Anthropic,
                       context: str,
                       company: dict) -> dict | None:
    """
    Tier 2: Claude Opus deep analysis on flagged items.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=DEEP_ANALYSIS_MODEL,
                max_tokens=4096,
                system=DEEP_ANALYSIS_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": context}
                ]
            )
            raw = response.content[0].text
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)

        except json.JSONDecodeError:
            # Return raw text in a wrapper if JSON parsing fails
            return {"raw_analysis": response.content[0].text}
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"      [!] Deep analysis failed: {e}")
                return None


# ---------------------------------------------------------------------------
# MAIN ANALYSIS LOOP
# ---------------------------------------------------------------------------

def analyse_company(identifier: str,
                     graph: GraphQuerier,
                     openai_client: OpenAI,
                     anthropic_client: anthropic.Anthropic,
                     output_dir: str) -> dict | None:
    """
    Runs the full two-tier analysis for a single company.
    Returns the findings dict or None if no material changes found.
    """
    print(f"\n  Analysing {identifier}...")

    # Get company metadata
    companies = graph.get_all_companies()
    company = next((c for c in companies if c["identifier"] == identifier), None)
    if not company:
        print(f"    [!] Company {identifier} not found in graph")
        return None

    # Get temporal pairs
    pairs = graph.get_temporal_pairs(identifier)
    if not pairs:
        print(f"    [!] No temporal pairs found for {identifier} - need at least 2 filings")
        return None

    # --- TIER 1: Screen all temporal pairs ---
    screened_flags = []
    total_pairs = 0

    for prev_id, curr_id, pair_meta in pairs:
        prev_rfs = graph.get_filing_risk_factors(prev_id)
        curr_rfs = graph.get_filing_risk_factors(curr_id)

        # Group by risk_category
        prev_by_cat = {}
        for rf in prev_rfs:
            cat = rf["risk_category"]
            prev_by_cat.setdefault(cat, []).append(rf["text"])

        curr_by_cat = {}
        for rf in curr_rfs:
            cat = rf["risk_category"]
            curr_by_cat.setdefault(cat, []).append(rf["text"])

        # Screen each shared category
        shared_cats = set(prev_by_cat.keys()) & set(curr_by_cat.keys())
        for cat in shared_cats:
            total_pairs += 1
            prev_text = " ".join(prev_by_cat[cat])
            curr_text = " ".join(curr_by_cat[cat])

            result = screen_temporal_change(
                openai_client, prev_text, curr_text, cat, identifier
            )

            if result and result.get("material_change") and \
               result.get("confidence", 0) >= SCREENING_THRESHOLD:
                screened_flags.append({
                    "risk_category":    cat,
                    "prev_period":      pair_meta["prev_period"],
                    "curr_period":      pair_meta["curr_period"],
                    "prev_filing_type": pair_meta["prev_type"],
                    "curr_filing_type": pair_meta["curr_type"],
                    "prev_text":        prev_text,
                    "curr_text":        curr_text,
                    "confidence":       result["confidence"],
                    "change_type":      result.get("change_type", "unknown"),
                    "screening_reason": result.get("reason", ""),
                })

    print(f"    Screened {total_pairs} category pairs  |  "
          f"{len(screened_flags)} flagged for deep analysis")

    if not screened_flags:
        return None

    # --- TIER 2: Claude Opus deep analysis ---
    print(f"    Running Claude Opus deep analysis...")
    context = build_deep_analysis_context(
        company, pairs, graph, screened_flags
    )
    analysis = run_deep_analysis(anthropic_client, context, company)

    if not analysis:
        return None

    # Assemble final findings
    findings = {
        "company":          company,
        "flags_count":      len(screened_flags),
        "flagged_items":    screened_flags,
        "deep_analysis":    analysis,
    }

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{identifier}_findings.json")
    with open(out_path, "w") as f:
        json.dump(findings, f, indent=2)

    signal = analysis.get("overall_signal", {}).get("signal_strength", "UNKNOWN")
    print(f"    Signal strength: {signal}  ->  saved to {out_path}")

    return findings


def main():
    parser = argparse.ArgumentParser(description="DDDS Graph-RAG Analysis Pipeline")
    parser.add_argument("--company",    default=None, help="Analyse a single company by identifier")
    parser.add_argument("--all",        action="store_true", help="Analyse all companies in graph")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for findings JSON files")
    parser.add_argument("--uri",        default=NEO4J_URI)
    parser.add_argument("--user",       default=NEO4J_USER)
    parser.add_argument("--password",   default=NEO4J_PASSWORD)
    args = parser.parse_args()

    if not args.company and not args.all:
        print("Specify --company IDENTIFIER or --all")
        return

    print(f"\n[DDDS] Graph-RAG Analysis Pipeline")
    print(f"  Screening model:  {SCREENING_MODEL}")
    print(f"  Deep analysis:    {DEEP_ANALYSIS_MODEL}")
    print(f"  Output dir:       {args.output_dir}\n")

    # Connect
    graph = GraphQuerier(args.uri, args.user, args.password)
    openai_client     = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    anthropic_client  = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Determine companies to analyse
    if args.company:
        targets = [args.company]
    else:
        companies = graph.get_all_companies()
        targets = [c["identifier"] for c in companies]
        print(f"  Analysing all {len(targets)} companies in graph\n")

    # Run analysis
    results = {"high": [], "medium": [], "low": [], "no_change": []}
    for identifier in targets:
        findings = analyse_company(
            identifier, graph, openai_client, anthropic_client, args.output_dir
        )
        if findings:
            signal = findings["deep_analysis"].get(
                "overall_signal", {}
            ).get("signal_strength", "LOW").upper()
            results.get(signal.lower(), results["low"]).append(identifier)
        else:
            results["no_change"].append(identifier)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Analysis Complete")
    print(f"{'='*50}")
    print(f"  HIGH signal:    {len(results['high'])} companies")
    print(f"  MEDIUM signal:  {len(results['medium'])} companies")
    print(f"  LOW signal:     {len(results['low'])} companies")
    print(f"  No change:      {len(results['no_change'])} companies")
    if results["high"]:
        print(f"\n  HIGH signal companies: {', '.join(results['high'])}")
    print(f"{'='*50}\n")

    graph.close()
    print("[Done] Findings saved to", args.output_dir)
    print("  Next: run generate_memos.py to produce investment memos")


if __name__ == "__main__":
    main()