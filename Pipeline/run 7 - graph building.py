"""
DDDS - Neo4j Graph Builder (AuraDB)
====================================
Reads topic_labelled.csv and peer_selections.csv to build a knowledge graph
optimised for period-over-period disclosure comparison.

Graph Schema
------------
Nodes:
    (:Company)          {ticker, sector}
    (:Filing)           {filing_id, ticker, filing_type, filing_date}
    (:Topic)            {name}
    (:DisclosureBlock)  {block_id, ticker, filing_date, filing_type,
                         section, topic, sentence_count, text}

Edges:
    (:Company)-[:HAS_FILING]->(:Filing)
    (:Filing)-[:HAS_DISCLOSURE]->(:DisclosureBlock)
    (:DisclosureBlock)-[:ABOUT_TOPIC]->(:Topic)
    (:Filing)-[:NEXT_PERIOD]->(:Filing)
    (:Company)-[:PEER_OF {distance, rank, sic_level}]->(:Company)

Key design decisions:
    - DisclosureBlock = all sentences for one (company, filing, section, topic)
      concatenated into a single text field. This is the unit of analysis
      for GPT-4o-mini period-over-period comparison.
    - Topic is a first-class node (not just a string property) so you can
      traverse Company -> Filing -> DisclosureBlock -> Topic across periods.
    - PEER_OF edges come from peer_selections.csv with distance/rank metadata.
    - 'other' topic sentences are excluded from the graph — they're boilerplate.
    - Temporal edges link filings by filing_date order per company.

Usage
-----
    python "run 7 - graph building.py"
    python "run 7 - graph building.py" --wipe
    python "run 7 - graph building.py" --input path/to/topic_labelled.csv

Pipeline position
-----------------
    run_0 (EDGAR download)
        -> run_1 (topic extraction -> topic_labelled.csv)
            -> run_5/6 (FinBERT training + prediction)
                -> **run_7 (Neo4j graph building)**  <- YOU ARE HERE
                    -> run_8 (Graph-RAG analysis)
"""

import argparse
import hashlib
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# CONFIG — reads from .env, falls back to AuraDB defaults
# ---------------------------------------------------------------------------
NEO4J_URI      = os.environ.get("NEO4J_URI", "neo4j+s://2e14ef0f.databases.neo4j.io")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

INPUT_CSV      = PROJECT_ROOT / "data/Graph Rag Creation Data/topic_labelled.csv"
PEERS_CSV      = PROJECT_ROOT / "data/Graph Rag Creation Data/peer_selections.csv"
BATCH_SIZE     = 500

VALID_TOPICS = {
    "liquidity_solvency",
    "debt_financing",
    "cost_margin_pressure",
    "supply_chain",
    "revenue_future_performance",
    "geopolitical_macro",
    "regulatory_legal",
    "operational_product",
    "cybersecurity_technology",
    "human_capital",
    "financial_sustainability",
    "accounting_audit",
}
# ---------------------------------------------------------------------------


def make_id(*parts) -> str:
    """Deterministic short ID from string parts."""
    return hashlib.md5("_".join(str(p) for p in parts).encode()).hexdigest()[:16]


def normalise_date(d: str) -> str:
    """
    Converts DD/MM/YYYY -> YYYY-MM-DD.
    Passes through anything already in YYYY-MM-DD format.
    """
    d = str(d).strip()
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", d)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return d


class GraphBuilder:
    """Manages Neo4j connection and all graph ingestion operations."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"  Connected to Neo4j at {uri}")

    def close(self):
        self.driver.close()

    def verify_connection(self):
        """Quick connectivity check — fails fast if credentials are wrong."""
        with self.driver.session() as session:
            session.run("RETURN 1 AS ok").single()
        print("  Connection verified")

    # ------------------------------------------------------------------
    # SCHEMA
    # ------------------------------------------------------------------

    def create_constraints(self):
        """Creates uniqueness constraints and indexes for all node types."""
        stmts = [
            "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
            "CREATE CONSTRAINT filing_id IF NOT EXISTS FOR (f:Filing) REQUIRE f.filing_id IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT block_id IF NOT EXISTS FOR (d:DisclosureBlock) REQUIRE d.block_id IS UNIQUE",
            "CREATE INDEX filing_ticker IF NOT EXISTS FOR (f:Filing) ON (f.ticker)",
            "CREATE INDEX filing_date IF NOT EXISTS FOR (f:Filing) ON (f.filing_date)",
            "CREATE INDEX block_topic IF NOT EXISTS FOR (d:DisclosureBlock) ON (d.topic)",
        ]
        with self.driver.session() as session:
            for s in stmts:
                try:
                    session.run(s)
                except Exception as e:
                    print(f"    [!] {e}")
        print("  Schema constraints and indexes applied")

    def wipe(self):
        """Deletes all nodes and relationships. Use with --wipe flag."""
        with self.driver.session() as session:
            # Batch delete to avoid memory issues on large graphs
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
                )
                deleted = result.single()["deleted"]
                if deleted == 0:
                    break
                print(f"    Deleted {deleted} nodes...")
        print("  Database wiped")

    # ------------------------------------------------------------------
    # TOPIC NODES (static — 12 topics)
    # ------------------------------------------------------------------

    def ingest_topics(self):
        query = """
        UNWIND $names AS name
        MERGE (:Topic {name: name})
        """
        with self.driver.session() as session:
            session.run(query, names=list(VALID_TOPICS))
        print(f"  Topics:            {len(VALID_TOPICS)} nodes")

    # ------------------------------------------------------------------
    # COMPANY NODES
    # ------------------------------------------------------------------

    def ingest_companies(self, df: pd.DataFrame):
        tickers = sorted(df["ticker"].unique().tolist())
        query = """
        UNWIND $rows AS row
        MERGE (c:Company {ticker: row.ticker})
        SET c.sector = 'US Industrials'
        """
        rows = [{"ticker": t} for t in tickers]
        with self.driver.session() as session:
            for i in range(0, len(rows), BATCH_SIZE):
                session.run(query, rows=rows[i:i + BATCH_SIZE])
        print(f"  Companies:         {len(tickers)} nodes")

    # ------------------------------------------------------------------
    # FILING NODES + HAS_FILING edges
    # ------------------------------------------------------------------

    def ingest_filings(self, df: pd.DataFrame):
        filings = (
            df.groupby(["ticker", "filing_type", "filing_date"])
            .size()
            .reset_index(name="_count")
        )
        filings["filing_id"] = filings.apply(
            lambda r: make_id(r["ticker"], r["filing_type"], r["filing_date"]),
            axis=1,
        )

        query = """
        UNWIND $rows AS row
        MERGE (f:Filing {filing_id: row.filing_id})
        SET f.ticker      = row.ticker,
            f.filing_type = row.filing_type,
            f.filing_date = row.filing_date
        WITH f, row
        MATCH (c:Company {ticker: row.ticker})
        MERGE (c)-[:HAS_FILING]->(f)
        """
        rows = filings[["filing_id", "ticker", "filing_type", "filing_date"]].to_dict("records")
        with self.driver.session() as session:
            for i in range(0, len(rows), BATCH_SIZE):
                session.run(query, rows=rows[i:i + BATCH_SIZE])

        print(f"  Filings:           {len(filings)} nodes")
        return filings

    # ------------------------------------------------------------------
    # DISCLOSURE BLOCK NODES + edges to Filing and Topic
    # ------------------------------------------------------------------

    def ingest_disclosure_blocks(self, df: pd.DataFrame):
        """
        Groups sentences by (ticker, filing_type, filing_date, section, topic_label)
        and concatenates text into one DisclosureBlock node.
        Excludes 'other' topic — those are boilerplate.
        """
        substantive = df[df["topic_label"].isin(VALID_TOPICS)].copy()

        grouped = (
            substantive.sort_values("sentence_num")
            .groupby(["ticker", "filing_type", "filing_date", "section", "topic_label"])
            .agg(
                text=("text", "\n".join),
                sentence_count=("text", "size"),
            )
            .reset_index()
        )

        grouped["block_id"] = grouped.apply(
            lambda r: make_id(
                r["ticker"], r["filing_type"], r["filing_date"],
                r["section"], r["topic_label"]
            ),
            axis=1,
        )
        grouped["filing_id"] = grouped.apply(
            lambda r: make_id(r["ticker"], r["filing_type"], r["filing_date"]),
            axis=1,
        )

        # --- Create DisclosureBlock nodes + link to Filing ---
        query_block = """
        UNWIND $rows AS row
        MERGE (d:DisclosureBlock {block_id: row.block_id})
        SET d.ticker         = row.ticker,
            d.filing_date    = row.filing_date,
            d.filing_type    = row.filing_type,
            d.section        = row.section,
            d.topic          = row.topic_label,
            d.sentence_count = row.sentence_count,
            d.text           = row.text
        WITH d, row
        MATCH (f:Filing {filing_id: row.filing_id})
        MERGE (f)-[:HAS_DISCLOSURE]->(d)
        """

        rows = grouped[[
            "block_id", "filing_id", "ticker", "filing_date", "filing_type",
            "section", "topic_label", "sentence_count", "text",
        ]].to_dict("records")

        with self.driver.session() as session:
            for i in range(0, len(rows), BATCH_SIZE):
                session.run(query_block, rows=rows[i:i + BATCH_SIZE])

        # --- Link DisclosureBlock -> Topic ---
        query_topic = """
        UNWIND $rows AS row
        MATCH (d:DisclosureBlock {block_id: row.block_id})
        MATCH (t:Topic {name: row.topic_label})
        MERGE (d)-[:ABOUT_TOPIC]->(t)
        """
        topic_rows = grouped[["block_id", "topic_label"]].to_dict("records")
        with self.driver.session() as session:
            for i in range(0, len(topic_rows), BATCH_SIZE):
                session.run(query_topic, rows=topic_rows[i:i + BATCH_SIZE])

        print(f"  DisclosureBlocks:  {len(grouped):,} nodes")

    # ------------------------------------------------------------------
    # TEMPORAL EDGES (NEXT_PERIOD)
    # ------------------------------------------------------------------

    def create_temporal_edges(self):
        """
        Links consecutive filings per company ordered by filing_date.
        YYYY-MM-DD strings sort lexicographically = chronologically.
        """
        query = """
        MATCH (c:Company)-[:HAS_FILING]->(f:Filing)
        WITH c, f ORDER BY f.filing_date ASC
        WITH c, collect(f) AS filings
        UNWIND range(0, size(filings)-2) AS i
        WITH filings[i] AS f1, filings[i+1] AS f2
        MERGE (f1)-[:NEXT_PERIOD]->(f2)
        """
        with self.driver.session() as session:
            result = session.run(query)
            summary = result.consume()
        print(f"  Temporal edges:    {summary.counters.relationships_created:,} NEXT_PERIOD created")

    # ------------------------------------------------------------------
    # PEER EDGES (from peer_selections.csv)
    # ------------------------------------------------------------------

    def create_peer_edges(self, peers_path: Path):
        """
        Creates directed PEER_OF edges from peer_selections.csv.
        Each row = one directed edge with distance, rank, sic_level metadata.
        Only creates edges between companies that exist in the graph.
        """
        if not peers_path.exists():
            print(f"  [!] {peers_path} not found — skipping peer edges")
            return

        peers = pd.read_csv(peers_path)
        required = {"target_ticker", "peer_ticker", "distance", "rank", "sic_level"}
        missing = required - set(peers.columns)
        if missing:
            print(f"  [!] peer_selections.csv missing columns: {missing}")
            return

        query = """
        UNWIND $rows AS row
        MATCH (c1:Company {ticker: row.target_ticker})
        MATCH (c2:Company {ticker: row.peer_ticker})
        MERGE (c1)-[r:PEER_OF]->(c2)
        SET r.distance  = row.distance,
            r.rank      = row.rank,
            r.sic_level = row.sic_level
        """
        rows = peers[["target_ticker", "peer_ticker", "distance", "rank", "sic_level"]].to_dict("records")

        created = 0
        with self.driver.session() as session:
            for i in range(0, len(rows), BATCH_SIZE):
                result = session.run(query, rows=rows[i:i + BATCH_SIZE])
                summary = result.consume()
                created += summary.counters.relationships_created

        print(f"  Peer edges:        {created:,} PEER_OF created (from {len(peers):,} rows)")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    def print_summary(self):
        queries = {
            "Companies":        "MATCH (c:Company) RETURN count(c) AS n",
            "Filings":          "MATCH (f:Filing) RETURN count(f) AS n",
            "Topics":           "MATCH (t:Topic) RETURN count(t) AS n",
            "DisclosureBlocks": "MATCH (d:DisclosureBlock) RETURN count(d) AS n",
            "HAS_FILING":       "MATCH ()-[r:HAS_FILING]->() RETURN count(r) AS n",
            "HAS_DISCLOSURE":   "MATCH ()-[r:HAS_DISCLOSURE]->() RETURN count(r) AS n",
            "ABOUT_TOPIC":      "MATCH ()-[r:ABOUT_TOPIC]->() RETURN count(r) AS n",
            "NEXT_PERIOD":      "MATCH ()-[r:NEXT_PERIOD]->() RETURN count(r) AS n",
            "PEER_OF":          "MATCH ()-[r:PEER_OF]->() RETURN count(r) AS n",
        }
        print(f"\n{'='*50}")
        print(f"  DDDS Graph Summary")
        print(f"{'='*50}")
        with self.driver.session() as session:
            total_nodes = 0
            total_rels = 0
            for label, q in queries.items():
                n = session.run(q).single()["n"]
                print(f"  {label:<25} {n:>8,}")
                if label in ("Companies", "Filings", "Topics", "DisclosureBlocks"):
                    total_nodes += n
                else:
                    total_rels += n
            print(f"  {'─'*35}")
            print(f"  {'Total nodes':<25} {total_nodes:>8,}")
            print(f"  {'Total relationships':<25} {total_rels:>8,}")
        print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_labelled_csv(path: Path) -> pd.DataFrame:
    """
    Loads topic_labelled.csv, validates columns, normalises dates,
    and prints a summary of what was loaded.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Expected output of run_1 (topic extraction) at this path."
        )

    df = pd.read_csv(path)

    required = {"ticker", "filing_type", "filing_date", "section",
                "sentence_num", "text", "topic_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", "ticker"])
    df["ticker"]      = df["ticker"].astype(str).str.strip()
    df["text"]        = df["text"].astype(str).str.strip()
    df["topic_label"] = df["topic_label"].astype(str).str.strip()
    df["filing_date"] = df["filing_date"].apply(normalise_date)

    # Summary
    substantive = df[df["topic_label"].isin(VALID_TOPICS)]
    print(f"  {len(df):,} total sentences loaded")
    print(f"  {df['ticker'].nunique()} companies")
    print(f"  {df.groupby(['ticker', 'filing_type', 'filing_date']).ngroups} filings")
    print(f"  {len(substantive):,} substantive sentences (excl. 'other')")

    # Topic distribution
    topic_counts = df["topic_label"].value_counts()
    for topic, count in topic_counts.items():
        marker = "  " if topic in VALID_TOPICS else "x "
        print(f"    {marker}{topic:<35} {count:>6,}")

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DDDS Neo4j Graph Builder (AuraDB)")
    parser.add_argument("--input",    type=Path, default=INPUT_CSV,
                        help="Path to topic_labelled.csv")
    parser.add_argument("--peers",    type=Path, default=PEERS_CSV,
                        help="Path to peer_selections.csv")
    parser.add_argument("--wipe",     action="store_true",
                        help="Wipe database before building")
    parser.add_argument("--uri",      default=NEO4J_URI)
    parser.add_argument("--user",     default=NEO4J_USER)
    parser.add_argument("--password", default=NEO4J_PASSWORD)
    args = parser.parse_args()

    if not args.password:
        raise ValueError(
            "NEO4J_PASSWORD not set. Add it to your .env file:\n"
            "  NEO4J_PASSWORD=<your_auradb_password>"
        )

    print(f"\n{'='*50}")
    print(f"  DDDS Neo4j Graph Builder (AuraDB)")
    print(f"{'='*50}")
    print(f"  Input:  {args.input}")
    print(f"  Peers:  {args.peers}")
    print(f"  Neo4j:  {args.uri}")
    print()

    # --- Step 1: Connect ---
    print("[1/8] Connecting to AuraDB...")
    builder = GraphBuilder(args.uri, args.user, args.password)
    builder.verify_connection()

    # --- Step 2: Wipe (optional) ---
    if args.wipe:
        print("\n  [!] --wipe flag set. Clearing database...")
        builder.wipe()

    # --- Step 3: Schema ---
    print("\n[2/8] Applying schema...")
    builder.create_constraints()

    # --- Step 4: Load CSV ---
    print("\n[3/8] Loading topic_labelled.csv...")
    df = load_labelled_csv(args.input)

    # --- Step 5: Topic nodes ---
    print("\n[4/8] Creating Topic nodes...")
    builder.ingest_topics()

    # --- Step 6: Company nodes ---
    print("\n[5/8] Creating Company nodes...")
    builder.ingest_companies(df)

    # --- Step 7: Filing nodes ---
    print("\n[6/8] Creating Filing nodes...")
    builder.ingest_filings(df)

    # --- Step 8: DisclosureBlock nodes ---
    print("\n[7/8] Creating DisclosureBlock nodes...")
    builder.ingest_disclosure_blocks(df)

    # --- Step 9: Edges ---
    print("\n[8/8] Creating edges...")
    builder.create_temporal_edges()
    builder.create_peer_edges(args.peers)

    # --- Summary ---
    builder.print_summary()
    builder.close()

    print("[Done] Graph built successfully.")
    print("  Next: run_8 (Graph-RAG analysis)\n")


if __name__ == "__main__":
    main()