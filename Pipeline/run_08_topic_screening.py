"""
DDDS - Disclosure Screening (Run 8)
=================================================
Pulls disclosure block pairs from the Neo4j graph and screens for
disclosure degradation using GPT-4.1-mini. Two comparison layers:

    Layer 1 — Temporal:  Same company, same filing type, same topic/section,
                         current vs prior period.
    Layer 2 — Peer:      Same topic/section, current filing vs each peer's
                         most recent filing of the same type.

Additionally detects "disappeared topics" — topics present in a prior
filing but absent in the current one — via pure graph traversal (no LLM).

Output: CSV of all flags for Opus escalation in run_9.

Usage
-----
    python "run 8 - graph RAG.py"               # full run
    python "run 8 - graph RAG.py" --test         # 5 random temporal pairs only
    python "run 8 - graph RAG.py" --no-cache     # ignore cached results
    python "run 8 - graph RAG.py" --concurrency 10

Pipeline position
-----------------
    run_7 (Neo4j graph building)
        → **run_8 (GPT-4o-mini screening)**  ← YOU ARE HERE
            → run_9 (Claude Opus escalation)
                → run_10 (memo generation)
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import AsyncOpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j+s://2e14ef0f.databases.neo4j.io")
NEO4J_USER     = os.environ.get("NEO4J_USER", "2e14ef0f")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

MODEL          = "gpt-4.1-mini"
CONCURRENCY    = 20
CACHE_DIR      = Path("data/Graph Rag Creation Data/screening_cache")
OUTPUT_CSV     = Path("data/Graph Rag Creation Data/screening_flags.csv")
# ---------------------------------------------------------------------------


TEMPORAL_SYSTEM_PROMPT = """You are a senior equity research analyst reviewing SEC filing disclosure changes for US Industrials companies. You have a HIGH BAR for flagging — most period-over-period changes are normal and expected. You are looking for genuinely concerning disclosure degradation that would warrant further investigation.

FLAG ONLY when you see MULTIPLE of the following occurring together:
- Specific dollar amounts, percentages, or timelines that were previously disclosed are now REMOVED (not updated — removed entirely)
- Concrete commitments or mitigation plans replaced with vague boilerplate language
- A previously detailed risk discussion collapsed into a brief generic statement
- Quantitative breakdowns consolidated into vague summaries with no replacement detail

DO NOT FLAG — these are NORMAL and expected:
- Numbers changing between periods (e.g. $30M becoming $25M) — that is an update, not degradation
- Risks being removed because they were resolved or are no longer relevant
- Language being reworded while preserving equivalent specificity
- Sections getting shorter because the underlying situation improved
- Different dollar figures, dates, or percentages replacing previous ones
- Adding qualifying language ("may", "could") alongside substantive detail
- Boilerplate disclaimers being added ON TOP OF existing substantive content
- Naturally evolving disclosure reflecting changed business conditions

The majority of comparisons you see should NOT be flagged. If you are unsure, do NOT flag. Only flag when you are confident a reasonable equity analyst would view this change as a deliberate reduction in transparency.

Respond with ONLY a JSON object:
{"flagged": true or false, "reason": "One sentence. Be specific about what was removed or obscured."}"""


PEER_SYSTEM_PROMPT = """You are a senior equity research analyst comparing how two peer companies in US Industrials discuss the same risk topic in their SEC filings. You have an EXTREMELY HIGH BAR for flagging.

These are two DIFFERENT COMPANIES. They have different business models, different exposures, different strategies, and different disclosure styles. All of this is NORMAL. Your job is NOT to find differences — differences are expected and inevitable.

FLAG ONLY when you see evidence that one company appears to be CONCEALING or DOWNPLAYING a risk that it is clearly exposed to:
- A company provides near-zero substance on a risk that is obviously material to its business (given what its close peer discloses about the same industry-wide risk)
- A company uses pure boilerplate or generic disclaimers where its peer reveals this is an active, quantifiable threat affecting the entire industry
- The asymmetry suggests deliberate evasion, not just different levels of exposure

DO NOT FLAG — ALL of these are NORMAL and expected:
- Different levels of detail — companies disclose differently
- Different levels of exposure — a company less affected by a risk naturally says less
- One company having more debt, more costs, more legal issues — that is a business difference, not a disclosure problem
- One company being more concise or using different structure
- One company quantifying a risk while another discusses it qualitatively
- Any difference that can be explained by the companies simply having different business situations

Ask yourself: "Does this look like one company is HIDING something, or do they just have different businesses?" If the answer is the latter, DO NOT FLAG.

Respond with ONLY a JSON object:
{"flagged": true or false, "reason": "One sentence. Explain what the company appears to be concealing and why."}"""


def make_cache_key(*parts) -> str:
    """Deterministic cache key from input parts."""
    return hashlib.md5("_".join(str(p) for p in parts).encode()).hexdigest()


class ScreeningCache:
    """Simple JSON file cache keyed by comparison inputs."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def set(self, key: str, value: dict):
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(value, f)

    def count(self) -> int:
        return len(list(self.cache_dir.glob("*.json")))


class GraphRetriever:
    """Pulls comparison pairs from Neo4j for LLM screening."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def verify(self):
        with self.driver.session() as session:
            session.run("RETURN 1").single()

    def get_temporal_pairs(self) -> list[dict]:
        """
        For each DisclosureBlock, find the same company's prior filing
        of the same type and match on topic + section.
        Returns pairs: {current_block, prior_block, metadata}.
        """
        query = """
        MATCH (c:Company)-[:HAS_FILING]->(f:Filing)
        WITH c, f
        ORDER BY f.filing_date DESC
        WITH c, collect(f)[0] AS current
        MATCH (c)-[:HAS_FILING]->(prior:Filing)
        WHERE prior.filing_type = current.filing_type
          AND prior.filing_date < current.filing_date
        WITH c, current, prior
        ORDER BY prior.filing_date DESC
        WITH c, current, collect(prior)[0] AS prior
        MATCH (current)-[:HAS_DISCLOSURE]->(d_curr:DisclosureBlock)
        MATCH (prior)-[:HAS_DISCLOSURE]->(d_prior:DisclosureBlock)
        WHERE d_prior.topic = d_curr.topic
          AND d_prior.section = d_curr.section
        RETURN c.ticker AS ticker,
               current.filing_type AS filing_type,
               current.filing_date AS current_date,
               prior.filing_date AS prior_date,
               d_curr.section AS section,
               d_curr.topic AS topic,
               d_curr.block_id AS current_block_id,
               d_prior.block_id AS prior_block_id,
               d_curr.text AS current_text,
               d_prior.text AS prior_text
        """
        with self.driver.session() as session:
            result = session.run(query)
            pairs = [dict(record) for record in result]
        return pairs

    def get_peer_pairs(self) -> list[dict]:
        """
        For each company's most recent filing, compare disclosure blocks
        against each peer's most recent filing of the same type,
        matched on topic + section.
        """
        query = """
        MATCH (c1:Company)-[peer:PEER_OF]->(c2:Company)
        MATCH (c1)-[:HAS_FILING]->(f1:Filing)
        WITH c1, c2, peer, f1
        ORDER BY f1.filing_date DESC
        WITH c1, c2, peer, collect(f1)[0] AS f1
        MATCH (c2)-[:HAS_FILING]->(f2:Filing)
        WHERE f2.filing_type = f1.filing_type
        WITH c1, c2, peer, f1, f2
        ORDER BY f2.filing_date DESC
        WITH c1, c2, peer, f1, collect(f2)[0] AS f2
        MATCH (f1)-[:HAS_DISCLOSURE]->(d1:DisclosureBlock)
        MATCH (f2)-[:HAS_DISCLOSURE]->(d2:DisclosureBlock)
        WHERE d1.topic = d2.topic AND d1.section = d2.section
        RETURN c1.ticker AS ticker,
               c2.ticker AS peer_ticker,
               peer.rank AS peer_rank,
               peer.distance AS peer_distance,
               f1.filing_type AS filing_type,
               f1.filing_date AS target_date,
               f2.filing_date AS peer_date,
               d1.section AS section,
               d1.topic AS topic,
               d1.block_id AS target_block_id,
               d2.block_id AS peer_block_id,
               d1.text AS target_text,
               d2.text AS peer_text
        """
        with self.driver.session() as session:
            result = session.run(query)
            pairs = [dict(record) for record in result]
        return pairs

    def get_disappeared_topics(self) -> list[dict]:
        """
        Finds topics present in a company's prior filing but absent
        in the current filing of the same type. No LLM needed.
        Only checks the most recent filing pair per type.
        """
        query = """
        MATCH (c:Company)-[:HAS_FILING]->(f:Filing)
        WITH c, f
        ORDER BY f.filing_date DESC
        WITH c, collect(f)[0] AS current
        MATCH (c)-[:HAS_FILING]->(prior:Filing)
        WHERE prior.filing_type = current.filing_type
          AND prior.filing_date < current.filing_date
        WITH c, current, prior
        ORDER BY prior.filing_date DESC
        WITH c, current, collect(prior)[0] AS prior
        MATCH (prior)-[:HAS_DISCLOSURE]->(d_prior:DisclosureBlock)
        OPTIONAL MATCH (current)-[:HAS_DISCLOSURE]->(d_curr:DisclosureBlock)
        WHERE d_curr.topic = d_prior.topic AND d_curr.section = d_prior.section
        WITH c, current, prior, d_prior, d_curr
        WHERE d_curr IS NULL
        RETURN c.ticker AS ticker,
               current.filing_type AS filing_type,
               current.filing_date AS current_date,
               prior.filing_date AS prior_date,
               d_prior.section AS section,
               d_prior.topic AS topic,
               d_prior.block_id AS prior_block_id
        """
        with self.driver.session() as session:
            result = session.run(query)
            disappeared = [dict(record) for record in result]
        return disappeared


class LLMScreener:
    """Async GPT-4o-mini screening with concurrency control and caching."""

    def __init__(self, api_key: str, model: str, concurrency: int, cache: ScreeningCache):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.cache = cache
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def screen_temporal(self, pair: dict, use_cache: bool = True) -> dict:
        """Screen a single temporal pair."""
        cache_key = make_cache_key("temporal", pair["current_block_id"], pair["prior_block_id"])

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        user_prompt = f"""TEMPORAL COMPARISON — {pair['ticker']} ({pair['filing_type']})
Topic: {pair['topic']}
Section: {pair['section']}

PRIOR PERIOD ({pair['prior_date']}):
{pair['prior_text']}

CURRENT PERIOD ({pair['current_date']}):
{pair['current_text']}

Has disclosure quality degraded from the prior period to the current period?"""

        result = await self._call_llm(user_prompt, TEMPORAL_SYSTEM_PROMPT)

        output = {
            "layer": "temporal",
            "ticker": pair["ticker"],
            "filing_type": pair["filing_type"],
            "section": pair["section"],
            "topic": pair["topic"],
            "current_date": pair["current_date"],
            "prior_date": pair["prior_date"],
            "current_block_id": pair["current_block_id"],
            "prior_block_id": pair["prior_block_id"],
            "flagged": result.get("flagged", False),
            "reason": result.get("reason", ""),
        }

        self.cache.set(cache_key, output)
        return output

    async def screen_peer(self, pair: dict, use_cache: bool = True) -> dict:
        """Screen a single peer pair."""
        cache_key = make_cache_key("peer", pair["target_block_id"], pair["peer_block_id"])

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        user_prompt = f"""PEER COMPARISON — {pair['ticker']} vs {pair['peer_ticker']} ({pair['filing_type']})
Topic: {pair['topic']}
Section: {pair['section']}

COMPANY A — {pair['ticker']} ({pair['target_date']}):
{pair['target_text']}

COMPANY B — {pair['peer_ticker']} ({pair['peer_date']}):
{pair['peer_text']}

Are these two peer companies treating this risk in materially different ways that would concern an equity analyst?"""

        result = await self._call_llm(user_prompt, PEER_SYSTEM_PROMPT)

        output = {
            "layer": "peer",
            "ticker": pair["ticker"],
            "peer_ticker": pair["peer_ticker"],
            "peer_rank": pair["peer_rank"],
            "peer_distance": pair["peer_distance"],
            "filing_type": pair["filing_type"],
            "section": pair["section"],
            "topic": pair["topic"],
            "target_date": pair["target_date"],
            "peer_date": pair["peer_date"],
            "target_block_id": pair["target_block_id"],
            "peer_block_id": pair["peer_block_id"],
            "flagged": result.get("flagged", False),
            "reason": result.get("reason", ""),
        }

        self.cache.set(cache_key, output)
        return output

    async def _call_llm(self, user_prompt: str, system_prompt: str) -> dict:
        """Make a single API call with retry logic."""
        async with self.semaphore:
            for attempt in range(3):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens

                    return json.loads(response.choices[0].message.content)

                except Exception as e:
                    if attempt < 2:
                        wait = 2 ** attempt
                        print(f"    [!] Retry {attempt+1}/3 after error: {e}")
                        await asyncio.sleep(wait)
                    else:
                        print(f"    [!] Failed after 3 attempts: {e}")
                        return {"flagged": False, "reason": f"ERROR: {str(e)[:100]}"}


async def run_screening(
    retriever: GraphRetriever,
    screener: LLMScreener,
    use_cache: bool = True,
    test_mode: bool = False,
) -> pd.DataFrame:
    """Main screening pipeline — retrieves pairs and screens them."""

    all_flags = []

    # ------------------------------------------------------------------
    # DISAPPEARED TOPICS (no LLM needed)
    # ------------------------------------------------------------------
    print("\n[1/3] Detecting disappeared topics...")
    disappeared = retriever.get_disappeared_topics()
    print(f"  {len(disappeared)} disappeared topic instances found")

    for d in disappeared:
        all_flags.append({
            "layer": "disappeared_topic",
            "ticker": d["ticker"],
            "filing_type": d["filing_type"],
            "section": d["section"],
            "topic": d["topic"],
            "current_date": d["current_date"],
            "prior_date": d["prior_date"],
            "prior_block_id": d["prior_block_id"],
            "flagged": True,
            "reason": f"Topic '{d['topic']}' in {d['section']} was present in {d['prior_date']} filing but absent in {d['current_date']}.",
            "peer_ticker": None,
            "peer_rank": None,
            "peer_distance": None,
            "target_date": d["current_date"],
            "peer_date": None,
            "current_block_id": None,
            "target_block_id": None,
            "peer_block_id": None,
        })

    # ------------------------------------------------------------------
    # TEMPORAL COMPARISONS
    # ------------------------------------------------------------------
    print("\n[2/3] Running temporal comparisons...")
    temporal_pairs = retriever.get_temporal_pairs()
    print(f"  {len(temporal_pairs)} temporal pairs retrieved from graph")

    if test_mode:
        # Sample 50 random companies, filter all pairs to those companies
        all_tickers = list({p["ticker"] for p in temporal_pairs})
        test_tickers = set(random.sample(all_tickers, min(50, len(all_tickers))))
        temporal_pairs = [p for p in temporal_pairs if p["ticker"] in test_tickers]
        print(f"  TEST MODE: {len(test_tickers)} companies selected, {len(temporal_pairs)} temporal pairs")

        # Also filter disappeared topics to test companies
        all_flags = [f for f in all_flags if f["ticker"] in test_tickers]

    start = time.time()
    temporal_tasks = [screener.screen_temporal(p, use_cache) for p in temporal_pairs]
    temporal_results = await asyncio.gather(*temporal_tasks)
    elapsed = time.time() - start

    temporal_flagged = sum(1 for r in temporal_results if r["flagged"])
    print(f"  Completed in {elapsed:.1f}s — {temporal_flagged}/{len(temporal_results)} flagged")
    all_flags.extend(temporal_results)

    # ------------------------------------------------------------------
    # PEER COMPARISONS
    # ------------------------------------------------------------------
    print("\n[3/3] Running peer comparisons...")
    peer_pairs = retriever.get_peer_pairs()
    print(f"  {len(peer_pairs)} total peer pairs retrieved from graph")

    # Limit to top 2 closest peers per company (by rank)
    peer_pairs = [p for p in peer_pairs if p["peer_rank"] <= 2]
    print(f"  {len(peer_pairs)} pairs after filtering to top 2 peers per company")

    if test_mode:
        peer_pairs = [p for p in peer_pairs if p["ticker"] in test_tickers]
        print(f"  TEST MODE: filtered to {len(peer_pairs)} peer pairs for test companies")

    start = time.time()
    peer_tasks = [screener.screen_peer(p, use_cache) for p in peer_pairs]
    peer_results = await asyncio.gather(*peer_tasks)
    elapsed = time.time() - start

    peer_flagged = sum(1 for r in peer_results if r["flagged"])
    print(f"  Completed in {elapsed:.1f}s — {peer_flagged}/{len(peer_results)} flagged")
    all_flags.extend(peer_results)

    # ------------------------------------------------------------------
    # BUILD OUTPUT
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_flags)
    return df


def print_test_results(df: pd.DataFrame):
    """Print per-company flag breakdown table for test mode."""
    flagged = df[df["flagged"] == True]

    if len(flagged) == 0:
        print("\n  No flags raised in test run.")
        return

    # Build per-company breakdown
    tickers = sorted(flagged["ticker"].unique())
    rows = []
    for ticker in tickers:
        t_df = flagged[flagged["ticker"] == ticker]
        temporal = len(t_df[t_df["layer"] == "temporal"])
        peer = len(t_df[t_df["layer"] == "peer"])
        disappeared = len(t_df[t_df["layer"] == "disappeared_topic"])
        total = temporal + peer + disappeared
        rows.append((ticker, temporal, peer, disappeared, total))

    rows.sort(key=lambda x: x[4], reverse=True)

    print(f"\n{'='*70}")
    print(f"  TEST RESULTS — Flag Breakdown by Company ({len(tickers)} companies)")
    print(f"{'='*70}")
    print(f"  {'Ticker':<10} {'Temporal':>10} {'Peer':>10} {'Disappeared':>13} {'Total':>8}")
    print(f"  {'─'*51}")

    for ticker, temporal, peer, disappeared, total in rows:
        print(f"  {ticker:<10} {temporal:>10} {peer:>10} {disappeared:>13} {total:>8}")

    print(f"  {'─'*51}")
    totals = (
        sum(r[1] for r in rows),
        sum(r[2] for r in rows),
        sum(r[3] for r in rows),
        sum(r[4] for r in rows),
    )
    print(f"  {'TOTAL':<10} {totals[0]:>10} {totals[1]:>10} {totals[2]:>13} {totals[3]:>8}")
    print(f"{'='*70}")

    # Show a few example flags from each layer
    for layer_name in ["temporal", "peer"]:
        layer_df = flagged[flagged["layer"] == layer_name]
        if len(layer_df) == 0:
            continue
        print(f"\n  Sample {layer_name} flags:")
        for _, row in layer_df.head(5).iterrows():
            peer_info = f" vs {row['peer_ticker']}" if pd.notna(row.get("peer_ticker")) else ""
            print(f"    {row['ticker']}{peer_info} | {row['topic']} | {row['section']}")
            print(f"      {row['reason']}")

    print()


def print_summary(df: pd.DataFrame, screener: LLMScreener):
    """Print final run summary with costs."""
    flagged = df[df["flagged"] == True]

    print(f"\n{'='*60}")
    print(f"  SCREENING SUMMARY")
    print(f"{'='*60}")

    # By layer
    for layer in ["temporal", "peer", "disappeared_topic"]:
        layer_df = df[df["layer"] == layer]
        layer_flagged = layer_df[layer_df["flagged"] == True]
        if len(layer_df) > 0:
            print(f"  {layer:<25} {len(layer_flagged):>5} / {len(layer_df):>6} flagged")

    print(f"  {'─'*45}")
    print(f"  {'Total':<25} {len(flagged):>5} / {len(df):>6} flagged")

    # Unique companies flagged
    flagged_tickers = flagged["ticker"].nunique()
    total_tickers = df["ticker"].nunique()
    print(f"  {'Companies flagged':<25} {flagged_tickers:>5} / {total_tickers:>6}")

    # Cost
    input_cost = screener.total_input_tokens * 0.70 / 1_000_000
    output_cost = screener.total_output_tokens * 2.80 / 1_000_000
    total_cost = input_cost + output_cost
    print(f"\n  API Usage:")
    print(f"    Input tokens:    {screener.total_input_tokens:>12,}")
    print(f"    Output tokens:   {screener.total_output_tokens:>12,}")
    print(f"    Input cost:      ${input_cost:>10.4f}")
    print(f"    Output cost:     ${output_cost:>10.4f}")
    print(f"    Total cost:      ${total_cost:>10.4f}")

    # Top flagged companies
    if len(flagged) > 0:
        print(f"\n  Top 10 most-flagged companies:")
        top = flagged.groupby("ticker").size().sort_values(ascending=False).head(10)
        for ticker, count in top.items():
            print(f"    {ticker:<10} {count:>4} flags")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="DDDS Disclosure Screening (Run 8)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 5 random temporal pairs only")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached results, re-run all comparisons")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"Max parallel API calls (default: {CONCURRENCY})")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    # Validate env
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-key":
        raise ValueError("OPENAI_API_KEY not set in .env")
    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD not set in .env")

    print(f"\n{'='*60}")
    print(f"  DDDS Disclosure Screening (GPT-4.1-mini)")
    print(f"{'='*60}")
    print(f"  Model:        {MODEL}")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  Cache:        {'disabled' if args.no_cache else 'enabled'}")
    print(f"  Test mode:    {args.test}")
    print(f"  Output:       {args.output}")
    print()

    # Connect to Neo4j
    print("[Setup] Connecting to AuraDB...")
    retriever = GraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    retriever.verify()
    print("  Connected")

    # Init screener
    cache = ScreeningCache(CACHE_DIR)
    print(f"  Cache contains {cache.count()} previous results")
    screener = LLMScreener(OPENAI_API_KEY, MODEL, args.concurrency, cache)

    # Run screening
    df = asyncio.run(run_screening(
        retriever=retriever,
        screener=screener,
        use_cache=not args.no_cache,
        test_mode=args.test,
    ))

    retriever.close()

    # Output
    if args.test:
        print_test_results(df)
    else:
        # Save full results
        df.to_csv(args.output, index=False)
        print(f"\n  Results saved to {args.output}")

        # Save flagged-only
        flagged_df = df[df["flagged"] == True]
        flagged_path = args.output.parent / "screening_flags_only.csv"
        flagged_df.to_csv(flagged_path, index=False)
        print(f"  Flagged-only saved to {flagged_path}")

        # Top 70% escalation — rank companies by flag count, escalate top 70%
        company_flags = flagged_df.groupby("ticker").size().reset_index(name="flag_count")
        company_flags = company_flags.sort_values("flag_count", ascending=False)
        cutoff_idx = int(len(company_flags) * 0.70)
        cutoff_count = company_flags.iloc[cutoff_idx - 1]["flag_count"] if cutoff_idx > 0 else 0

        escalated_tickers = company_flags.head(cutoff_idx)["ticker"].tolist()
        dropped_tickers = company_flags.tail(len(company_flags) - cutoff_idx)["ticker"].tolist()

        # Save escalated flags for run_9
        escalated_df = flagged_df[flagged_df["ticker"].isin(escalated_tickers)]
        escalated_path = args.output.parent / "screening_escalated.csv"
        escalated_df.to_csv(escalated_path, index=False)

        print(f"\n  Escalation cutoff: top 70% = {len(escalated_tickers)} companies (>= {cutoff_count} flags)")
        print(f"  Dropped: {len(dropped_tickers)} companies below threshold")
        print(f"  Escalated flags saved to {escalated_path}")

    print_summary(df, screener)

    if args.test:
        print("  Run without --test to execute full screening.")
    else:
        print("[Done] Screening complete.")
        print("  Next: run_9 (Claude Opus escalation)\n")


if __name__ == "__main__":
    main()