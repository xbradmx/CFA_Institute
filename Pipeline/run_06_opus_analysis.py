"""
DDDS - Claude Opus Escalation (Run 6)
======================================
Reads the GPT-4.1-mini screening flags from Run 5, retrieves the
underlying filing text from Neo4j via block_id lookups, and sends
each company's flag bundle to Claude Opus for deep analysis.

Opus returns short-form reasoning per flag:
    - Whether the flag represents a genuine disclosure degradation signal
    - Brief justification citing specific language from the passages
    - Signal strength rating (HIGH / MEDIUM / LOW / DISMISS)

Outputs are cached per ticker to avoid redundant API calls.
Run 07 consumes these outputs to generate investment memos.

Usage
-----
    python run_06_opus_analysis.py --all
    python run_06_opus_analysis.py --ticker NTAP
    python run_06_opus_analysis.py --all --use-cached
    python run_06_opus_analysis.py --all --dry-run
"""

import argparse
import csv
import json
import os
import random
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import anthropic
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
load_dotenv()

NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j+s://2e14ef0f.databases.neo4j.io")
NEO4J_USER     = os.environ.get("NEO4J_USER", "2e14ef0f")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

OPUS_MODEL      = "claude-opus-4-6"
MAX_TOKENS      = 8192
RETRY_DELAY     = 60    # wait 60s on rate limit before retry
CONCURRENCY     = 2     # max 2 parallel requests — never more

INPUT_CSV       = "data/Graph Rag Creation Data/screening_escalated.csv"
CACHE_DIR       = "data/Graph Rag Creation Data/opus_cache"
OUTPUT_DIR      = "data/Graph Rag Creation Data/opus_analysis"

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NEO4J BLOCK RETRIEVAL
# ---------------------------------------------------------------------------

class GraphQuerier:
    """Minimal Neo4j interface for block_id text lookups."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_blocks_batch(self, block_ids: list[str]) -> dict[str, str]:
        """Batch retrieval of block texts. Returns {block_id: text}."""
        query = """
        UNWIND $ids AS bid
        MATCH (d:DisclosureBlock {block_id: bid})
        RETURN d.block_id AS block_id, d.text AS text
        """
        with self.driver.session() as session:
            results = session.run(query, ids=block_ids)
            return {r["block_id"]: r["text"] for r in results}

    def get_company_metadata(self, ticker: str) -> dict | None:
        """Returns company node metadata."""
        query = """
        MATCH (c:Company {ticker: $ticker})
        RETURN c.ticker AS ticker, c.name AS name, c.sector AS sector
        """
        with self.driver.session() as session:
            result = session.run(query, ticker=ticker)
            record = result.single()
            return dict(record) if record else None


# ---------------------------------------------------------------------------
# DATA LOADING & GROUPING
# ---------------------------------------------------------------------------

def load_screening_flags(csv_path: str) -> dict[str, list[dict]]:
    """
    Loads screening_escalated.csv and groups flags by ticker.
    Returns {ticker: [flag_dicts]}.
    """
    flags_by_ticker: dict[str, list[dict]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"].strip()
            if not ticker:
                continue
            flags_by_ticker[ticker].append(row)

    return dict(flags_by_ticker)


def collect_block_ids(flags: list[dict]) -> list[str]:
    """Extracts all non-empty block_ids from a company's flags."""
    id_cols = ["prior_block_id", "current_block_id",
               "target_block_id", "peer_block_id"]
    ids = set()
    for flag in flags:
        for col in id_cols:
            val = flag.get(col, "").strip()
            if val:
                ids.add(val)
    return list(ids)


# ---------------------------------------------------------------------------
# CONTEXT BUILDING
# ---------------------------------------------------------------------------

def build_opus_context(ticker: str,
                       flags: list[dict],
                       block_texts: dict[str, str],
                       company_meta: dict | None) -> str:
    """
    Assembles the structured context payload for Opus Pass 1.
    Groups flags by layer type for clarity.
    """
    sections = []

    # Header
    name = company_meta["name"] if company_meta else ticker
    sector = company_meta.get("sector", "Unknown") if company_meta else "Unknown"
    sections.append(f"COMPANY: {name} ({ticker})")
    sections.append(f"SECTOR: {sector}")
    sections.append(f"TOTAL FLAGS: {len(flags)}")
    sections.append("")

    # Group by layer
    by_layer: dict[str, list[dict]] = defaultdict(list)
    for flag in flags:
        by_layer[flag["layer"]].append(flag)

    flag_index = 0

    # --- Disappeared topics ---
    if "disappeared_topic" in by_layer:
        sections.append("=" * 60)
        sections.append("DISAPPEARED TOPICS")
        sections.append("Topics present in a prior filing but absent in the current one.")
        sections.append("=" * 60)

        for flag in by_layer["disappeared_topic"]:
            flag_index += 1
            prior_text = block_texts.get(flag["prior_block_id"], "[text not found]")
            sections.append(f"\n--- FLAG {flag_index} ---")
            sections.append(f"Topic: {flag['topic']}")
            sections.append(f"Section: {flag['section']}")
            sections.append(f"Filing type: {flag['filing_type']}")
            sections.append(f"Prior date: {flag['prior_date']}")
            sections.append(f"Current date: {flag['current_date']}")
            sections.append(f"Block ID (prior): {flag['prior_block_id']}")
            sections.append(f"Screening reason: {flag['reason']}")
            sections.append(f"\nPRIOR PERIOD TEXT (now absent):")
            sections.append(prior_text)
            sections.append("")

    # --- Temporal changes ---
    if "temporal" in by_layer:
        sections.append("=" * 60)
        sections.append("TEMPORAL CHANGES")
        sections.append("Same topic present in both periods but language changed materially.")
        sections.append("=" * 60)

        for flag in by_layer["temporal"]:
            flag_index += 1
            prior_text = block_texts.get(flag["prior_block_id"], "[text not found]")
            curr_text = block_texts.get(flag["current_block_id"], "[text not found]")
            sections.append(f"\n--- FLAG {flag_index} ---")
            sections.append(f"Topic: {flag['topic']}")
            sections.append(f"Section: {flag['section']}")
            sections.append(f"Filing type: {flag['filing_type']}")
            sections.append(f"Prior date: {flag['prior_date']} -> Current date: {flag['current_date']}")
            sections.append(f"Block ID (prior): {flag['prior_block_id']}")
            sections.append(f"Block ID (current): {flag['current_block_id']}")
            sections.append(f"Screening reason: {flag['reason']}")
            sections.append(f"\nPRIOR PERIOD:")
            sections.append(prior_text)
            sections.append(f"\nCURRENT PERIOD:")
            sections.append(curr_text)
            sections.append("")

    # --- Peer divergence ---
    if "peer" in by_layer:
        sections.append("=" * 60)
        sections.append("PEER DIVERGENCE")
        sections.append("This company's disclosure diverges from a peer on the same topic.")
        sections.append("=" * 60)

        for flag in by_layer["peer"]:
            flag_index += 1
            target_text = block_texts.get(flag["target_block_id"], "[text not found]")
            peer_text = block_texts.get(flag["peer_block_id"], "[text not found]")
            sections.append(f"\n--- FLAG {flag_index} ---")
            sections.append(f"Topic: {flag['topic']}")
            sections.append(f"Section: {flag['section']}")
            sections.append(f"Peer: {flag['peer_ticker']} (rank {flag['peer_rank']}, "
                           f"distance {flag['peer_distance']})")
            sections.append(f"Target date: {flag['target_date']} | "
                           f"Peer date: {flag['peer_date']}")
            sections.append(f"Block ID (target): {flag['target_block_id']}")
            sections.append(f"Block ID (peer): {flag['peer_block_id']}")
            sections.append(f"Screening reason: {flag['reason']}")
            sections.append(f"\nTARGET COMPANY ({ticker}):")
            sections.append(target_text)
            sections.append(f"\nPEER COMPANY ({flag['peer_ticker']}):")
            sections.append(peer_text)
            sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# OPUS PROMPT
# ---------------------------------------------------------------------------

OPUS_SYSTEM_PROMPT = """\
You are a senior buy-side equity analyst evaluating whether flagged \
disclosure changes represent genuine disclosure degradation signals.

You are part of the Disclosure Degradation Detection System (DDDS), a \
supply chain monitoring tool that helps analysts detect when upstream \
suppliers are becoming less transparent in their SEC filings. The flags \
you are reviewing were generated by a GPT-4o-mini screening layer and \
need your expert judgement.

For each numbered flag in the context, provide:
1. Your assessment: GENUINE, BORDERLINE, or DISMISS
2. The block_ids exactly as provided (these are sentence identifiers \
used downstream to trace findings back to specific filing passages)
3. Two to three sentences of reasoning, citing specific language from \
the passages
4. If GENUINE or BORDERLINE, what an analyst monitoring this supplier \
should investigate

A flag is GENUINE if:
- Specific metrics, dollar amounts, or quantitative detail were removed \
and replaced with vague language
- A material risk category disappeared entirely without explanation
- Language was softened or made deliberately ambiguous on a topic where \
peers are more specific
- The change pattern is consistent with obfuscation rather than \
legitimate editorial revision

A flag should be DISMISSED if:
- The change is routine boilerplate revision with no substantive difference
- The topic was consolidated elsewhere in the filing rather than removed
- Peer divergence reflects genuine business differences rather than \
disclosure evasion
- The screening reason overstates a minor rewording

After analysing all flags, provide an OVERALL COMPANY ASSESSMENT with:
- Overall signal strength: HIGH / MEDIUM / LOW
- A two to three sentence summary of the disclosure quality picture
- The top three concerns an analyst should prioritise

Respond ONLY with valid JSON in this structure:
{
  "flags": [
    {
      "flag_number": 1,
      "topic": "...",
      "layer": "disappeared_topic|temporal|peer",
      "block_ids": {
        "prior": "block_id or null",
        "current": "block_id or null",
        "target": "block_id or null",
        "peer": "block_id or null"
      },
      "assessment": "GENUINE|BORDERLINE|DISMISS",
      "reasoning": "...",
      "investigation_action": "..." or null
    }
  ],
  "overall": {
    "signal_strength": "HIGH|MEDIUM|LOW",
    "summary": "...",
    "top_concerns": ["...", "...", "..."],
    "genuine_count": 0,
    "borderline_count": 0,
    "dismissed_count": 0
  }
}"""


# ---------------------------------------------------------------------------
# OPUS API CALL (bulletproof — never crashes, retries forever on rate limits)
# ---------------------------------------------------------------------------

def call_opus(client: anthropic.Anthropic,
              context: str,
              ticker: str) -> dict | None:
    """
    Calls Claude Opus. Retries indefinitely on rate limits (60s backoff).
    Only gives up after 3 consecutive non-rate-limit errors.
    """
    non_rate_failures = 0

    while True:
        try:
            response = client.messages.create(
                model=OPUS_MODEL,
                max_tokens=MAX_TOKENS,
                system=OPUS_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": context}
                ]
            )

            raw = response.content[0].text
            raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                print(f"\n      [!] {ticker}: JSON parse failed, storing raw")
                result = {"raw_analysis": raw}

            result["_meta"] = {
                "ticker": ticker,
                "model": OPUS_MODEL,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = ("429" in str(e) or "rate" in err_str or
                           "overloaded" in err_str or "capacity" in err_str)

            if is_rate_limit:
                # Rate limit — wait 60s and retry, never give up
                wait = RETRY_DELAY + random.uniform(1, 10)
                print(f"\n      [!] {ticker}: Rate limited, "
                      f"retrying in {wait:.0f}s...", flush=True)
                time.sleep(wait)
                continue
            else:
                non_rate_failures += 1
                if non_rate_failures >= 3:
                    print(f"\n      [!] {ticker}: 3 non-rate-limit failures, "
                          f"giving up. Last error: {e}")
                    return None
                wait = 15 * non_rate_failures
                print(f"\n      [!] {ticker}: Error ({non_rate_failures}/3): "
                      f"{e}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
                continue


# ---------------------------------------------------------------------------
# CACHING
# ---------------------------------------------------------------------------

def get_cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{ticker}_opus.json")


def load_cached(ticker: str, cache_dir: str) -> dict | None:
    path = get_cache_path(ticker, cache_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_cache(ticker: str, data: dict, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = get_cache_path(ticker, cache_dir)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# SINGLE TICKER PROCESSING
# ---------------------------------------------------------------------------

def process_ticker(ticker: str,
                   flags: list[dict],
                   block_texts: dict[str, str],
                   company_meta: dict | None,
                   client: anthropic.Anthropic,
                   cache_dir: str,
                   output_dir: str,
                   use_cached: bool = False,
                   dry_run: bool = False) -> dict | None:
    """Processes a single ticker through Opus escalation."""

    # Check cache
    if use_cached:
        cached = load_cached(ticker, cache_dir)
        if cached:
            print(f"    [cached] {ticker}")
            return cached

    # Build context (block_texts already pre-fetched)
    context = build_opus_context(ticker, flags, block_texts, company_meta)

    if dry_run:
        est_tokens = len(context) // 4
        print(f"    [dry-run] {ticker}: {len(flags)} flags, "
              f"~{len(context):,} chars, ~{est_tokens:,} tokens")
        return {"_meta": {"input_tokens": est_tokens, "output_tokens": 0}}

    # Call Opus
    print(f"    Calling Opus for {ticker} ({len(flags)} flags)...",
          end=" ", flush=True)
    result = call_opus(client, context, ticker)

    if result:
        save_cache(ticker, result, cache_dir)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{ticker}_opus_analysis.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        signal = result.get("overall", {}).get("signal_strength", "?")
        genuine = result.get("overall", {}).get("genuine_count", "?")
        tokens_in = result.get("_meta", {}).get("input_tokens", 0)
        tokens_out = result.get("_meta", {}).get("output_tokens", 0)
        print(f"done  {signal} signal | {genuine} genuine | "
              f"{tokens_in}+{tokens_out} tokens")
    else:
        print("FAILED")

    return result


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DDDS Opus Escalation (Pass 1)"
    )
    parser.add_argument("--ticker", default=None,
                        help="Process a single ticker")
    parser.add_argument("--all", action="store_true",
                        help="Process all tickers in screening_escalated.csv")
    parser.add_argument("--test", action="store_true",
                        help="Process 5 random tickers that have both "
                             "temporal and peer flags")
    parser.add_argument("--input", default=INPUT_CSV,
                        help="Path to screening_escalated.csv")
    parser.add_argument("--cache-dir", default=CACHE_DIR,
                        help="Directory for cached Opus responses")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory for analysis JSON outputs")
    parser.add_argument("--use-cached", action="store_true",
                        help="Skip tickers with existing cached responses")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build contexts without calling Opus (cost check)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"Parallel Opus requests (default: {CONCURRENCY})")
    parser.add_argument("--neo4j-uri", default=NEO4J_URI)
    parser.add_argument("--neo4j-user", default=NEO4J_USER)
    parser.add_argument("--neo4j-password", default=NEO4J_PASSWORD)
    args = parser.parse_args()

    if not args.ticker and not args.all and not args.test:
        print("Specify --ticker TICKER, --all, or --test")
        return

    print(f"\n[DDDS] Opus Escalation — Pass 1")
    print(f"  Input:      {args.input}")
    print(f"  Cache dir:  {args.cache_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Model:      {OPUS_MODEL}")
    print(f"  Concurrency: {args.concurrency}")
    if args.test:
        print(f"  Mode:       test (5 random tickers with peer+temporal)")
    if args.use_cached:
        print(f"  Mode:       use-cached (skip existing)")
    if args.dry_run:
        print(f"  Mode:       dry-run (no API calls)")

    flags_by_ticker = load_screening_flags(args.input)
    print(f"  Loaded {sum(len(v) for v in flags_by_ticker.values())} flags "
          f"across {len(flags_by_ticker)} tickers\n")

    # Determine targets
    if args.ticker:
        if args.ticker not in flags_by_ticker:
            print(f"  [!] Ticker {args.ticker} not found in screening flags")
            return
        targets = [args.ticker]
    elif args.test:
        # Pick 5 random tickers that have BOTH temporal and peer flags
        eligible = []
        for t, flags in flags_by_ticker.items():
            layers = {f["layer"] for f in flags}
            if "temporal" in layers and "peer" in layers:
                eligible.append(t)
        if len(eligible) < 5:
            print(f"  [!] Only {len(eligible)} tickers have both temporal "
                  f"and peer flags — using all of them")
            targets = eligible
        else:
            targets = sorted(random.sample(eligible, 5))
        print(f"  Test tickers: {', '.join(targets)}\n")
    else:
        targets = sorted(flags_by_ticker.keys())

    # Connect
    graph = GraphQuerier(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Pre-fetch all block texts from Neo4j in one batch
    print(f"  Pre-fetching block texts from Neo4j...")
    all_block_ids = []
    for t in targets:
        all_block_ids.extend(collect_block_ids(flags_by_ticker[t]))
    all_block_ids = list(set(all_block_ids))

    all_block_texts = graph.get_blocks_batch(all_block_ids) if all_block_ids else {}
    missing = len(all_block_ids) - len(all_block_texts)
    print(f"  Fetched {len(all_block_texts)}/{len(all_block_ids)} blocks"
          + (f" ({missing} missing)" if missing else ""))

    # Block text verification
    text_lengths = [len(v) for v in all_block_texts.values()]
    if text_lengths:
        avg_len = sum(text_lengths) / len(text_lengths)
        max_len = max(text_lengths)
        min_len = min(text_lengths)
        print(f"  Block text stats: avg {avg_len:,.0f} chars, "
              f"min {min_len:,}, max {max_len:,}")

    # Pre-fetch company metadata
    all_company_meta = {}
    for t in targets:
        meta = graph.get_company_metadata(t)
        if meta:
            all_company_meta[t] = meta

    graph.close()
    print(f"  Neo4j pre-fetch complete, connection closed\n")

    # Process — max 2 concurrent with ThreadPoolExecutor
    results: dict[str, list[str]] = {
        "HIGH": [], "MEDIUM": [], "LOW": [], "FAILED": []
    }
    total_input_tokens = 0
    total_output_tokens = 0
    completed = 0
    results_lock = threading.Lock()

    concurrency = min(args.concurrency, CONCURRENCY)
    if args.dry_run:
        concurrency = 1

    def _process_one(ticker: str) -> tuple[str, dict | None]:
        return ticker, process_ticker(
            ticker=ticker,
            flags=flags_by_ticker[ticker],
            block_texts=all_block_texts,
            company_meta=all_company_meta.get(ticker),
            client=client,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            use_cached=args.use_cached,
            dry_run=args.dry_run,
        )

    print(f"  Processing {len(targets)} tickers "
          f"({concurrency} concurrent)...\n")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_process_one, t): t for t in targets}

        for future in as_completed(futures):
            ticker, result = future.result()

            with results_lock:
                completed += 1
                if result:
                    signal = result.get("overall", {}).get(
                        "signal_strength", "LOW"
                    ).upper()
                    bucket = signal if signal in results else "LOW"
                    results[bucket].append(ticker)
                    total_input_tokens += result.get("_meta", {}).get(
                        "input_tokens", 0
                    )
                    total_output_tokens += result.get("_meta", {}).get(
                        "output_tokens", 0
                    )
                elif not args.dry_run:
                    results["FAILED"].append(ticker)

                # Progress update every 10 completions
                if completed % 10 == 0 or completed == len(targets):
                    print(f"\n  --- Progress: {completed}/{len(targets)} "
                          f"complete ---", flush=True)

    # Summary
    print(f"\n{'=' * 55}")
    print(f"  Opus Escalation Summary")
    print(f"{'=' * 55}")
    if not args.dry_run:
        print(f"  HIGH:     {len(results['HIGH'])} companies")
        print(f"  MEDIUM:   {len(results['MEDIUM'])} companies")
        print(f"  LOW:      {len(results['LOW'])} companies")
        print(f"  FAILED:   {len(results['FAILED'])} companies")
        print(f"\n  Total tokens: {total_input_tokens:,} in + "
              f"{total_output_tokens:,} out")
        est_cost = (total_input_tokens * 15 / 1_000_000) + \
                   (total_output_tokens * 75 / 1_000_000)
        print(f"  Estimated cost: ${est_cost:.2f}")
        if results["HIGH"]:
            print(f"\n  HIGH signal: {', '.join(results['HIGH'][:15])}")
    else:
        print(f"  Dry run complete — {len(targets)} tickers checked")

    print(f"{'=' * 55}")
    print(f"\n[Done] Next: run 9 consumes {args.output_dir}/")


if __name__ == "__main__":
    main()