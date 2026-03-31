# CFA_Institute
for the cfa challenge ai 




# DDDS Extraction Pipeline

Scans a flat folder of SEC filings and earnings call transcripts and
consolidates extracted sections into a single passages CSV ready for
the Stage 3 labelling pipeline.

---

## Setup

```bash
pip install pdfplumber beautifulsoup4 pandas
```

---

## Usage

```bash
python extract_sections.py --input-dir data/filings
python extract_sections.py --input-dir data/filings --output data/passages.csv
```

---

## Supported formats

| Format | Parser |
|---|---|
| `.pdf` | pdfplumber |
| `.html` / `.htm` | BeautifulSoup4 |
| `.txt` | Built-in (tries utf-8, latin-1, cp1252) |

---

## What gets extracted

| Filing type | Sections extracted |
|---|---|
| 10-K | MD&A, Risk Factors |
| 10-Q | MD&A, Risk Factors |
| Earnings transcript | Full transcript text |
| unknown | MD&A and Risk Factors attempted |

Filing type is detected automatically from document content (cover page
and headers). No reliance on filename convention.

---

## Output CSV columns

| Column | Description |
|---|---|
| `filename` | Source file name |
| `company` | Detected company name (from cover page) |
| `identifier` | Detected ticker or CIK |
| `filing_type` | 10-K, 10-Q, earnings_transcript, or unknown |
| `section` | MD&A, Risk Factors, or Transcript |
| `risk_category` | financing, liquidity and solvency, costs, future performance, financial sustainability, product quality, or general |
| `text` | Extracted passage text |

---

## Risk category assignment

Each passage is assigned a risk category by keyword matching against
the six DDDS categories. The category with the most keyword hits wins.
Passages with no keyword matches are assigned `general`.

The keyword lists are defined in `RISK_CATEGORY_KEYWORDS` at the top
of the script and can be extended if your filings use sector-specific
terminology not currently covered.

---

## Tuning parameters

Defined at the top of `extract_sections.py`:

| Parameter | Default | Effect |
|---|---|---|
| `MIN_CHARS` | 200 | Passages shorter than this are dropped |
| `MAX_CHARS` | 8000 | Sections longer than this are split on paragraph boundaries |

---

## Feeding into Stage 3

Once extraction is complete:

```bash
python ../labelling_pipeline/label_vagueness.py  --input data/passages.csv
python ../labelling_pipeline/label_complexity.py --input data/passages.csv
```

---

## Known limitations

- Company name and ticker detection relies on standard SEC cover page
  formatting. Non-standard or heavily formatted PDFs may return empty
  strings for `company` and `identifier`. These can be filled in manually
  or via a post-processing step without affecting the labelling pipeline,
  which only requires `text` and `risk_category`.

- Section boundary detection uses heading pattern matching. Heavily
  nested HTML filings or PDFs with unusual formatting may cause sections
  to be missed or truncated. Check the extraction summary for files
  reporting 0 passages and inspect those manually.

- Earnings call transcripts are not section-segmented — the full text
  is treated as a single passage type. Risk category is assigned by
  keyword matching as normal.



# DDDS Stage 4 - Graph-RAG Pipeline

Two scripts covering graph ingestion and the two-tier LLM analysis layer.

---

## Scripts

| Script | Purpose |
|---|---|
| `build_graph.py` | Ingests passages CSV into Neo4j, creates all nodes and edges |
| `graph_rag.py` | Runs GPT-4o-mini screening and Claude Opus deep analysis |

---

## Setup

### 1. Install Neo4j Desktop
Download from https://neo4j.com/download/

1. Create a new project
2. Add a local DBMS (Neo4j 5.x)
3. Set a password
4. Start the DBMS
5. Connection: `bolt://localhost:7687`

### 2. Install Python dependencies
```bash
pip install neo4j anthropic openai tiktoken
```

### 3. Set environment variables
```bash
export NEO4J_PASSWORD='your-neo4j-password'
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
```

---

## Usage

### Step 1 — Build the graph
```bash
python build_graph.py --input data/passages.csv
```

First run on a fresh database:
```bash
python build_graph.py --input data/passages.csv --wipe
```

The script will print a graph summary on completion:
```
  Companies              266
  Filings              1,064
  RiskFactors         12,800
  NEXT_PERIOD          1,064
  PEER_OF             35,245
  SHARES_TOPIC        48,312
```

### Step 2 — Run analysis

Single company:
```bash
python graph_rag.py --company AAPL
```

All companies:
```bash
python graph_rag.py --all
```

Custom output directory:
```bash
python graph_rag.py --all --output-dir data/findings
```

---

## Graph Schema

### Nodes

| Node | Key Properties |
|---|---|
| `Company` | `identifier`, `name`, `sector` |
| `Filing` | `filing_id`, `company_id`, `filing_type`, `period`, `filename` |
| `RiskFactor` | `rf_id`, `filing_id`, `section`, `risk_category`, `text`, `vague_label`, `vague_prob`, `complex_label`, `complex_prob` |

### Edges

| Edge | Connects | Meaning |
|---|---|---|
| `HAS_FILING` | Company → Filing | Company owns this filing |
| `HAS_RISK_FACTOR` | Filing → RiskFactor | Filing contains this passage |
| `NEXT_PERIOD` | Filing → Filing | Temporal successor (same company) |
| `PEER_OF` | Company ↔ Company | Same sector peer |
| `SHARES_TOPIC` | RiskFactor ↔ RiskFactor | Same risk category, same period, different company |

---

## Two-Tier LLM Architecture

### Tier 1 — GPT-4o-mini screening
- Compares each risk category passage across consecutive filing periods
- Returns: `material_change`, `confidence`, `change_type`, `reason`
- Items with `material_change=true` and `confidence >= 0.7` are escalated

### Tier 2 — Claude Opus deep analysis
Receives all flagged items for a company and produces:
- **Temporal analysis**: exact language changes, metric removal, deterioration assessment
- **Contradiction analysis**: 10-K vs earnings call discrepancies
- **Peer analysis**: deviation from sector norms, omitted risks
- **Overall signal**: HIGH / MEDIUM / LOW with analyst action items

---

## Output Format

Each analysed company produces a JSON file at `data/findings/<identifier>_findings.json`:

```json
{
  "company": {"identifier": "...", "name": "...", "sector": "..."},
  "flags_count": 3,
  "flagged_items": [...],
  "deep_analysis": {
    "temporal_analysis": {...},
    "contradiction_analysis": {...},
    "peer_analysis": {...},
    "overall_signal": {
      "signal_strength": "HIGH",
      "summary": "...",
      "analyst_actions": [...]
    }
  }
}
```

---

## Adding FinBERT scores to the graph

Once Run 5 and Run 6 are complete, re-run the predictions on `data/passages.csv`:

```bash
python ../vagueness_classifier/predict.py \
    --model outputs/vagueness_model \
    --input data/passages.csv \
    --output data/passages_scored.csv

python ../complexity_classifier/predict.py \
    --model outputs/complexity_model \
    --input data/passages_scored.csv \
    --output data/passages_scored.csv
```

Then re-ingest with the scored CSV:
```bash
python build_graph.py --input data/passages_scored.csv --wipe
```

The `vague_label`, `vague_prob`, `complex_label`, `complex_prob` columns will populate the RiskFactor nodes, enabling score-based peer comparison in the graph queries.

---

## Pipeline position

```
Run 1  extract_sections.py  →  data/passages.csv
                                     ↓
                            build_graph.py  →  Neo4j graph
                                     ↓
                             graph_rag.py  →  data/findings/*.json
                                     ↓
                          generate_memos.py  →  investment memos  (Stage 5)
```