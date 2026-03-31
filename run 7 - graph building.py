"""
DDDS - Neo4j Graph Schema & Ingestion Pipeline
===============================================
Reads the passages CSV produced by Run 1 (extract_sections.py) and
populates a Neo4j knowledge graph with Company, Filing, and RiskFactor
nodes connected by temporal, peer, and risk-topic edges.

Graph Schema
------------
Nodes:
    (:Company)      {identifier, name, sector}
    (:Filing)       {filing_id, company_id, filing_type, period, filename}
    (:RiskFactor)   {rf_id, filing_id, section, risk_category, text,
                     vague_label, vague_prob, complex_label, complex_prob}

Edges:
    (:Company)-[:HAS_FILING]->(:Filing)
    (:Filing)-[:HAS_RISK_FACTOR]->(:RiskFactor)
    (:Filing)-[:NEXT_PERIOD]->(:Filing)          # temporal edge
    (:Company)-[:PEER_OF]->(:Company)            # peer edge (same sector)
    (:RiskFactor)-[:SHARES_TOPIC]->(:RiskFactor) # risk-topic edge (same category)

Usage
-----
    python build_graph.py --input data/passages.csv
    python build_graph.py --input data/passages.csv --wipe
"""

import argparse
import os
import hashlib
import pandas as pd
from neo4j import GraphDatabase
from itertools import combinations
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
load_dotenv()

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD") # TTPPassword!#
INPUT_CSV      = "data/passages.csv"
BATCH_SIZE     = 500    # rows per transaction batch
# ---------------------------------------------------------------------------


def make_id(*parts) -> str:
    """Deterministic ID from concatenated string parts."""
    return hashlib.md5("_".join(str(p) for p in parts).encode()).hexdigest()[:16]


class GraphBuilder:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"  Connected to Neo4j at {uri}")

    def close(self):
        self.driver.close()

    def verify_connection(self):
        with self.driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            result.single()

    # -----------------------------------------------------------------------
    # SCHEMA
    # -----------------------------------------------------------------------

    def create_constraints(self):
        """
        Creates uniqueness constraints and indexes.
        Must be run before ingestion.
        """
        constraints = [
            "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.identifier IS UNIQUE",
            "CREATE CONSTRAINT filing_id IF NOT EXISTS FOR (f:Filing) REQUIRE f.filing_id IS UNIQUE",
            "CREATE CONSTRAINT rf_id IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.rf_id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector)",
            "CREATE INDEX rf_category IF NOT EXISTS FOR (r:RiskFactor) ON (r.risk_category)",
            "CREATE INDEX filing_period IF NOT EXISTS FOR (f:Filing) ON (f.period)",
            "CREATE INDEX filing_company IF NOT EXISTS FOR (f:Filing) ON (f.company_id)",
        ]
        with self.driver.session() as session:
            for stmt in constraints + indexes:
                try:
                    session.run(stmt)
                except Exception as e:
                    print(f"    [!] Schema statement skipped: {e}")
        print("  Schema constraints and indexes applied")

    def wipe_database(self):
        """Deletes all nodes and relationships. Use with caution."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("  Database wiped")

    # -----------------------------------------------------------------------
    # NODE INGESTION
    # -----------------------------------------------------------------------

    def ingest_companies(self, df: pd.DataFrame):
        """
        Creates Company nodes from unique identifiers.
        Uses sector derived from risk_category distribution as a proxy
        until SEC sector metadata is available.
        """
        companies = df.groupby("identifier").agg(
            name=("company", "first"),
        ).reset_index()

        query = """
        UNWIND $rows AS row
        MERGE (c:Company {identifier: row.identifier})
        SET c.name   = row.name,
            c.sector = 'US Industrials'
        """
        with self.driver.session() as session:
            session.run(query, rows=companies.to_dict("records"))

        print(f"  Companies:    {len(companies)} nodes")

    def ingest_filings(self, df: pd.DataFrame):
        """
        Creates Filing nodes. Each unique (identifier, filename) pair
        is treated as one filing. Period is extracted from filename where
        possible, otherwise left as 'unknown'.
        """
        filings = df.groupby(["identifier", "filename"]).agg(
            filing_type=("filing_type", "first"),
        ).reset_index()

        filings["filing_id"] = filings.apply(
            lambda r: make_id(r["identifier"], r["filename"]), axis=1
        )
        filings["period"] = filings["filename"].apply(self._extract_period)

        query = """
        UNWIND $rows AS row
        MERGE (f:Filing {filing_id: row.filing_id})
        SET f.company_id  = row.identifier,
            f.filename    = row.filename,
            f.filing_type = row.filing_type,
            f.period      = row.period
        WITH f, row
        MATCH (c:Company {identifier: row.identifier})
        MERGE (c)-[:HAS_FILING]->(f)
        """
        with self.driver.session() as session:
            session.run(query, rows=filings.to_dict("records"))

        print(f"  Filings:      {len(filings)} nodes")
        return filings

    def ingest_risk_factors(self, df: pd.DataFrame):
        """
        Creates RiskFactor nodes from passage rows.
        Includes FinBERT classifier outputs if present.
        """
        df = df.copy()
        df["rf_id"] = df.apply(
            lambda r: make_id(r["identifier"], r["filename"], r.name), axis=1
        )
        df["filing_id"] = df.apply(
            lambda r: make_id(r["identifier"], r["filename"]), axis=1
        )

        # Optional classifier columns — default to None if not yet run
        for col in ["vague_label", "vague_prob", "complex_label", "complex_prob"]:
            if col not in df.columns:
                df[col] = None

        query = """
        UNWIND $rows AS row
        MERGE (r:RiskFactor {rf_id: row.rf_id})
        SET r.filing_id     = row.filing_id,
            r.section       = row.section,
            r.risk_category = row.risk_category,
            r.text          = row.text,
            r.vague_label   = row.vague_label,
            r.vague_prob    = row.vague_prob,
            r.complex_label = row.complex_label,
            r.complex_prob  = row.complex_prob
        WITH r, row
        MATCH (f:Filing {filing_id: row.filing_id})
        MERGE (f)-[:HAS_RISK_FACTOR]->(r)
        """
        cols = ["rf_id", "filing_id", "section", "risk_category", "text",
                "vague_label", "vague_prob", "complex_label", "complex_prob"]

        rows = df[cols].where(pd.notnull(df[cols]), None).to_dict("records")

        with self.driver.session() as session:
            for i in range(0, len(rows), BATCH_SIZE):
                session.run(query, rows=rows[i:i + BATCH_SIZE])

        print(f"  RiskFactors:  {len(df)} nodes")

    # -----------------------------------------------------------------------
    # EDGE CREATION
    # -----------------------------------------------------------------------

    def create_temporal_edges(self, filings: pd.DataFrame):
        """
        Creates NEXT_PERIOD edges between consecutive filings for
        the same company, ordered by period string (lexicographic).
        Works for YYYY, YYYYQ1 style period strings.
        """
        query = """
        MATCH (c:Company)-[:HAS_FILING]->(f:Filing)
        WITH c, f ORDER BY f.period ASC
        WITH c, collect(f) AS filings
        UNWIND range(0, size(filings)-2) AS i
        WITH filings[i] AS f1, filings[i+1] AS f2
        MERGE (f1)-[:NEXT_PERIOD]->(f2)
        """
        with self.driver.session() as session:
            session.run(query)
        print("  Temporal edges:    NEXT_PERIOD created")

    def create_peer_edges(self):
        """
        Creates PEER_OF edges between all companies in the same sector.
        Relationship is bidirectional — created once per pair.
        """
        query = """
        MATCH (c1:Company), (c2:Company)
        WHERE c1.sector = c2.sector
          AND c1.identifier < c2.identifier
        MERGE (c1)-[:PEER_OF]->(c2)
        MERGE (c2)-[:PEER_OF]->(c1)
        """
        with self.driver.session() as session:
            session.run(query)
        print("  Peer edges:        PEER_OF created")

    def create_risk_topic_edges(self):
        """
        Creates SHARES_TOPIC edges between RiskFactor nodes that share
        the same risk_category. Limited to within the same filing period
        to avoid creating an unmanageably dense graph.
        Edges connect risk factors from different companies only.
        """
        query = """
        MATCH (r1:RiskFactor)<-[:HAS_RISK_FACTOR]-(f1:Filing)<-[:HAS_FILING]-(c1:Company)
        MATCH (r2:RiskFactor)<-[:HAS_RISK_FACTOR]-(f2:Filing)<-[:HAS_FILING]-(c2:Company)
        WHERE r1.risk_category = r2.risk_category
          AND f1.period = f2.period
          AND c1.identifier < c2.identifier
          AND r1.rf_id < r2.rf_id
        MERGE (r1)-[:SHARES_TOPIC]->(r2)
        """
        with self.driver.session() as session:
            session.run(query)
        print("  Risk-topic edges:  SHARES_TOPIC created")

    # -----------------------------------------------------------------------
    # UTILITIES
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_period(filename: str) -> str:
        """
        Attempts to extract a period string from a filename.
        Falls back to 'unknown' if no recognisable pattern found.
        Examples: '2024Q3', '2024', 'unknown'
        """
        import re
        # Match YYYY Q patterns
        m = re.search(r"(20\d{2})[_\-\s]?[Qq]?([1-4])?", filename)
        if m:
            year = m.group(1)
            quarter = m.group(2)
            return f"{year}Q{quarter}" if quarter else year
        return "unknown"

    def print_graph_summary(self):
        queries = {
            "Companies":   "MATCH (c:Company) RETURN count(c) AS n",
            "Filings":     "MATCH (f:Filing) RETURN count(f) AS n",
            "RiskFactors": "MATCH (r:RiskFactor) RETURN count(r) AS n",
            "HAS_FILING":  "MATCH ()-[r:HAS_FILING]->() RETURN count(r) AS n",
            "HAS_RF":      "MATCH ()-[r:HAS_RISK_FACTOR]->() RETURN count(r) AS n",
            "NEXT_PERIOD": "MATCH ()-[r:NEXT_PERIOD]->() RETURN count(r) AS n",
            "PEER_OF":     "MATCH ()-[r:PEER_OF]->() RETURN count(r) AS n",
            "SHARES_TOPIC":"MATCH ()-[r:SHARES_TOPIC]->() RETURN count(r) AS n",
        }
        print(f"\n{'='*45}")
        print(f"  Graph Summary")
        print(f"{'='*45}")
        with self.driver.session() as session:
            for label, q in queries.items():
                n = session.run(q).single()["n"]
                print(f"  {label:<20} {n:>8,}")
        print(f"{'='*45}\n")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "risk_category", "identifier", "filename",
                "filing_type", "section", "company"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {missing}\n"
            f"Found: {list(df.columns)}\n"
            f"Make sure you are using the output of run_1_-_risk_factor_script.py"
        )
    df = df.dropna(subset=["text", "identifier"])
    df["identifier"] = df["identifier"].astype(str).str.strip()
    df["filename"]   = df["filename"].astype(str).str.strip()
    df["text"]       = df["text"].astype(str).str.strip()
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="DDDS Neo4j Graph Ingestion")
    parser.add_argument("--input",    default=INPUT_CSV, help="Path to passages CSV")
    parser.add_argument("--wipe",     action="store_true", help="Wipe database before ingestion")
    parser.add_argument("--uri",      default=NEO4J_URI)
    parser.add_argument("--user",     default=NEO4J_USER)
    parser.add_argument("--password", default=NEO4J_PASSWORD)
    args = parser.parse_args()

    print(f"\n[DDDS] Neo4j Graph Ingestion")
    print(f"  Input:    {args.input}")
    print(f"  Neo4j:    {args.uri}\n")

    print("[1/7] Connecting to Neo4j...")
    builder = GraphBuilder(args.uri, args.user, args.password)
    builder.verify_connection()

    if args.wipe:
        print("\n  [!] --wipe flag set. Deleting all existing data...")
        builder.wipe_database()

    print("\n[2/7] Applying schema constraints...")
    builder.create_constraints()

    print("\n[3/7] Loading CSV...")
    df = load_csv(args.input)
    print(f"  {len(df)} passages loaded")
    print(f"  {df['identifier'].nunique()} unique companies")
    print(f"  {df['filename'].nunique()} unique filings")

    print("\n[4/7] Ingesting Company nodes...")
    builder.ingest_companies(df)

    print("\n[5/7] Ingesting Filing nodes...")
    filings = builder.ingest_filings(df)

    print("\n[6/7] Ingesting RiskFactor nodes...")
    builder.ingest_risk_factors(df)

    print("\n[7/7] Creating edges...")
    builder.create_temporal_edges(filings)
    builder.create_peer_edges()
    builder.create_risk_topic_edges()

    builder.print_graph_summary()
    builder.close()

    print("[Done] Graph built successfully.")
    print("  Next: run graph_rag.py to start the analysis pipeline")


if __name__ == "__main__":
    main()