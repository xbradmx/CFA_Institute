# DDDS — Disclosure Degradation Detection System

**CFA AI Investment Challenge 2026**  
**Team:** The Transparency Project — Lancaster University  
**Members:** Brad McCann, Connor O'Keeffe, Ebro Dossajee

---

## Overview

DDDS automatically reads 10-K and 10-Q filings for 185 US Industrials companies (SIC 3400-3599), scores each sentence for vagueness and linguistic complexity using fine-tuned FinBERT classifiers, and flags companies whose disclosure quality is deteriorating relative to their own history and sector peers. Flagged companies receive a formatted two-page investment memo with sentence-level citations, quarter-on-quarter comparisons, and peer context.

The system is organised into two pipelines: a one-time **training pipeline** that fine-tunes the FinBERT classifiers, and a **scoring pipeline** that applies them to the live filing universe.

---

## Repository Structure

```
CFA_Institute/
├── Pipeline/                        # Scoring pipeline (Runs 0-11)
│   ├── run 0 - sec data collection for LLM.py
│   ├── run 0a - Clean Amendments.py
│   ├── run 1 - Topic Extraction.py
│   ├── run 2 - peer selection.py
│   ├── run 6 - vague_pred.py
│   ├── run 6 - complex_pred.py
│   ├── run 7 - graph building.py
│   ├── run 8 - topic screening.py
│   ├── run 9 - creating final memo.py
│   ├── run 10 - risk heat map.py
│   ├── run 11 - backtesting.py
│   ├── cache.py
│   └── resumebatches.py
│
├── training/                        # Training pipeline (one-time)
│   ├── run 0 - sec data collection for finbert.py
│   ├── run 2 - vague labelling.py
│   ├── run 2 - complex labelling.py
│   ├── run 3 - human agreement checker.py
│   ├── run 3a - generate validation sheet.py
│   ├── run 3b - merging human labels.py
│   ├── run 4 - TTV script.py
│   ├── run 5 - vague_training.py
│   └── run 5 - training_complex.py
│
├── data/
│   ├── Filings by company/          # Raw HTML filings (Run 0 output)
│   ├── training_sentences/          # Extracted training sentences
│   ├── Graph Rag Creation Data/     # topic_labelled.csv, peer_selections.csv
│   ├── training/                    # Train/val/test splits per classifier
│   ├── analyst_outputs/             # Per-filing DDDS score CSVs
│   └── validation/                  # Human annotation sheet
│
├── models/
│   ├── finbert_vagueness/           # Fine-tuned vagueness classifier
│   └── finbert_complexity/          # Fine-tuned complexity classifier
│
├── outputs/
│   ├── vagueness_model/             # Training checkpoints (vagueness)
│   └── complexity_model/            # Training checkpoints (complexity)
│
├── RUN USER BACKEND - vague_complex_analysis_pipeline.py
├── RUN USER FRONTEND - vague_complex_analysis_pipeline.py
├── RUN USER - regional heatmap - analyst outputs.py
├── requirements.txt
├── .env                             # Not committed — see Setup below
└── README.md
```

---

## Prerequisites

- Python 3.11
- [Neo4j Desktop](https://neo4j.com/download/) with a local DBMS running on `neo4j://127.0.0.1:7687`
- An OpenAI API key with Batch API access
- An Anthropic API key

---

## Setup

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Create the `.env` file

Create a file named `.env` in the project root. It is excluded from version control via `.gitignore`. Populate it as follows:

```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD="your-neo4j-password"
EDGAR_USER_AGENT=DDDS Research your-email@lancaster.ac.uk
FINDINGS_DIR=data/findings
ANALYST_OUTPUTS_DIR=data/analyst_outputs
```

Note: if your Neo4j password contains special characters, wrap it in double quotes as shown.

### 3. Configure Neo4j Desktop

1. Download and install Neo4j Desktop from https://neo4j.com/download/
2. Create a new project and add a local DBMS (Neo4j 5.x)
3. Set a password matching the one in your `.env` file
4. Start the DBMS before running any graph pipeline steps

---

## Reproducing Results Without Re-Running Labelling

The repository includes cached GPT-4o Batch API labelling results in JSONL format (`batch_69d66adcee948190a330ef4adc361dcd_output.jsonl`). All downstream outputs can be reproduced from this cache without re-submitting any batch jobs or incurring additional API cost. To use cached results, pass `--use-cached` to the relevant scripts as noted in the steps below.

---

## Training Pipeline

The training pipeline is a one-time process that produces the two fine-tuned FinBERT classifiers. Pre-trained model files are included in `models/finbert_vagueness/` and `models/finbert_complexity/`. Run this pipeline only if you need to retrain from scratch.

All scripts below are in the `training/` folder. Run them from the project root.

---

### Training Run 0 — Collect Training Corpus

Downloads one 10-K and one 10-Q per company from SIC 3400-3599 filed before June 2023, providing a minimum 16-month separation from the scoring window to prevent data leakage.

```powershell
python "training/run 0 - sec data collection for finbert.py"
```

Output: `data/training_sentences/training_sentences.csv`

---

### Training Run 2 — GPT-4o Consensus Labelling

Submits two independent Batch API jobs per classifier. Only sentences where both runs agree are carried forward. Run both vagueness and complexity labelling.

```powershell
python "training/run 2 - vague labelling.py"
python "training/run 2 - complex labelling.py"
```

Batch jobs are asynchronous. Monitor progress with:

```powershell
python "training/run 2 - vague labelling.py" --mode status
```

If a batch was interrupted, resume from where it left off:

```powershell
python "Pipeline/resumebatches.py"
```

To replay from the included cache without re-submitting:

```powershell
python "training/run 2 - vague labelling.py" --use-cached
python "training/run 2 - complex labelling.py" --use-cached
```

Output: `data/labelled/vagueness/labelled.csv`, `data/labelled/complexity/labelled.csv`

---

### Training Run 3a — Generate Validation Sheet

Stratifies 300 consensus sentences per classifier (150 per class) into a formatted Excel annotation sheet for human review. The sheet is split across three annotators at 100 sentences each.

```powershell
python "training/run 3a - generate validation sheet.py"
```

Output: `data/validation/validation_annotation_sheet.xlsx`

**Manual step:** each annotator labels their 100 sentences independently and returns the completed sheet. The three completed sheets are then merged before proceeding to Run 3b.

---

### Training Run 3b — Merge Human Labels

Merges the three completed annotation sheets into a single labelled CSV and applies the `human_validated` flag used in downstream stratification.

```powershell
python "training/run 3b - merging human labels.py"
```

---

### Training Run 3 — Inter-Annotator Agreement Check

Computes Cohen's Kappa for all three pairwise annotator combinations and for GPT-4o against the human majority vote. A Kappa of at least 0.9 across all pairings is required before proceeding.

```powershell
python "training/run 3 - human agreement checker.py"
```

---

### Training Run 4 — Train/Val/Test Split

Constructs a 3,000-sentence dataset per classifier. All 300 human-validated rows are guaranteed to be included. The remaining 2,700 are stratified-sampled from the GPT-labelled pool. Applies a 70/15/15 split stratified jointly on label and `human_validated`.

```powershell
python "training/run 4 - TTV script.py"
```

Output: `data/training/vagueness/train.csv`, `val.csv`, `test.csv`  
Output: `data/training/complexity/train.csv`, `val.csv`, `test.csv`

---

### Training Run 5 — Fine-Tune FinBERT

Fine-tunes two binary classifiers independently from the finbert checkpoint. Training uses weighted cross-entropy for class imbalance, macro F1 as the best-model selection criterion, and early stopping with patience of 2.

```powershell
python "training/run 5 - vague_training.py"
python "training/run 5 - training_complex.py"
```

Output: `outputs/vagueness_model/`, `outputs/complexity_model/`

Final model weights are also copied to `models/finbert_vagueness/` and `models/finbert_complexity/` for use in the scoring pipeline.

---

## Scoring Pipeline

The scoring pipeline applies the trained classifiers to the live filing universe (October 2024 to September 2025), builds the knowledge graph, runs the two-tier LLM analysis, and generates all outputs. All scripts below are in the `Pipeline/` folder unless otherwise noted. Run them from the project root.

---

### Run 0 — Collect Live Filings

Downloads the 2 most recent 10-K filings and 3 most recent 10-Q filings per company in SIC 3400-3599 within the October 2024 to September 2025 window. Amendments are automatically excluded. The company universe is cached after the first run.

```powershell
python "Pipeline/run 0 - sec data collection for LLM.py"
```

On subsequent runs, reuse the cached universe to skip the ~19-minute rebuild:

```powershell
python "Pipeline/run 0 - sec data collection for LLM.py" --use-cached
```

Output: `data/Filings by company/{TICKER}/10-K/` and `10-Q/`  
Output: `company_filing_summary.xlsx`

---

### Run 0a — Clean Amendments

Removes any amendment files (10-K-A, 10-Q-A) and enforces that every company folder contains exactly 2 x 10-K and 3 x 10-Q. Companies that fail this check are removed from the universe.

```powershell
python "Pipeline/run 0a - Clean Amendments.py" --data-dir "data/Filings by company"
```

To preview changes without deleting anything:

```powershell
python "Pipeline/run 0a - Clean Amendments.py" --data-dir "data/Filings by company" --dry-run
```

---

### Run 1 — Topic Extraction

Extracts Risk Factors and MD&A sections from each HTML filing, splits them into sentences, and classifies each sentence into one of 12 topic codes using GPT-4o-mini via the Batch API.

Run in sequence:

```powershell
# Step 1: Extract sentences from all filings
python "Pipeline/run 1 - Topic Extraction.py" --mode extract

# Step 2: Submit Batch API job for topic labelling
python "Pipeline/run 1 - Topic Extraction.py" --mode batch

# Step 3: Monitor until complete (auto-polls every 60 seconds)
python "Pipeline/run 1 - Topic Extraction.py" --mode watch

# Step 4: Download results and merge into final CSV
python "Pipeline/run 1 - Topic Extraction.py" --mode download
```

To use the included cached results and skip the batch submission entirely:

```powershell
python "Pipeline/run 1 - Topic Extraction.py" --mode download --use-cached
```

Output: `data/Graph Rag Creation Data/topic_labelled.csv`

---

### Run 2 — Peer Selection

Constructs a point-in-time peer group for each company filing using Euclidean distance across three normalised financial metrics: market capitalisation, trailing twelve-month revenue, and total assets. Peer candidates are identified via tiered SIC grouping (4-digit first, falling back to 3-digit then 2-digit). The five closest peers are written to CSV and later ingested into the Neo4j graph as `PEER_OF` edges.

```powershell
python "Pipeline/run 2 - peer selection.py" --input-dir "data/Filings by company"
```

On subsequent runs, reuse cached financial data from yfinance:

```powershell
python "Pipeline/run 2 - peer selection.py" --use-cached
```

Output: `data/Graph Rag Creation Data/peer_selections.csv`

---

### Run 6 — FinBERT Inference

Runs batch inference with the fine-tuned classifiers on all sentences in `topic_labelled.csv`. Both classifiers must be run.

```powershell
python "Pipeline/run 6 - vague_pred.py" \
    --model models/finbert_vagueness \
    --input "data/Graph Rag Creation Data/topic_labelled.csv" \
    --output data/predictions_vagueness.csv

python "Pipeline/run 6 - complex_pred.py" \
    --model models/finbert_complexity \
    --input "data/Graph Rag Creation Data/topic_labelled.csv" \
    --output data/predictions_complexity.csv
```

Output: `data/predictions_vagueness.csv`, `data/predictions_complexity.csv`

---

### Run 7 — Build Neo4j Graph

Ingests `topic_labelled.csv` and `peer_selections.csv` into the Neo4j knowledge graph. Ensure Neo4j Desktop is running before executing.

Graph schema:

| Node | Key properties |
|---|---|
| `Company` | ticker, sector |
| `Filing` | filing_id, ticker, filing_type, filing_date |
| `Topic` | name |
| `DisclosureBlock` | block_id, ticker, filing_date, section, topic, text |

| Edge | Meaning |
|---|---|
| `HAS_FILING` | Company owns filing |
| `HAS_DISCLOSURE` | Filing contains disclosure block |
| `ABOUT_TOPIC` | Block belongs to topic |
| `NEXT_PERIOD` | Temporal successor within same company |
| `PEER_OF` | Peer relationship with distance and SIC-level metadata |

```powershell
python "Pipeline/run 7 - graph building.py"
```

To wipe the database and rebuild from scratch:

```powershell
python "Pipeline/run 7 - graph building.py" --wipe
```

---

### Run 8 — Disclosure Screening (Graph-RAG Tier 1)

Queries the Neo4j graph for disclosure block pairs and screens them for degradation using GPT-4.1-mini. Two comparison layers are applied: temporal (same company, current vs prior period) and peer (same topic, current company vs each of its five peers). Disappeared topics are detected via graph traversal without any LLM call. Flagged results are written to CSV for escalation in Run 9.

```powershell
python "Pipeline/run 8 - topic screening.py"
```

To run a quick test on five random temporal pairs before the full run:

```powershell
python "Pipeline/run 8 - topic screening.py" --test
```

To use the LLM response cache and avoid repeated API calls on re-runs:

```powershell
python "Pipeline/run 8 - topic screening.py" --use-cached
```

Output: `data/Graph Rag Creation Data/screening_flags.csv`

---

### Run 9 — Investment Memo Generation (Graph-RAG Tier 2)

Passes all flagged items per company to Claude Opus for deep analysis covering temporal deterioration, peer comparison, and cross-document contradictions. Generates a formatted two-page `.docx` investment memo per flagged company. All outputs include model provenance and confidence levels.

```powershell
python "Pipeline/run 9 - creating final memo.py"
```

Output: `data/memos/{TICKER}_memo.docx`

---

### Run 10 — Geographic Risk Heatmap

Produces a global heatmap attributing raw vagueness scores to countries based on geographic keyword matching within each scored sentence. The USA receives scores from all sentences across the universe. No demeaning is applied to this output.

```powershell
python "Pipeline/run 10 - risk heat map.py"
```

To export a PNG without opening the interactive window:

```powershell
python "Pipeline/run 10 - risk heat map.py" --save outputs/heatmap.png --headless
```

Output: `outputs/heatmap.png`

---

### Run 11 — Backtesting

Tests the core hypothesis using the Li (2008) earnings persistence regression. Flagged companies are expected to show lower earnings persistence in the period following the flag relative to unflagged peers. The regression is run across three subsamples: full sample, profitable companies only (earnings > 0), and loss-making companies only (earnings < 0). Earnings data is pulled from the SEC EDGAR XBRL API with a yfinance fallback.

```powershell
python "Pipeline/run 11 - backtesting.py"
```

Optional arguments:

```powershell
# Set minimum signal level to count as flagged (HIGH, MEDIUM, or LOW)
python "Pipeline/run 11 - backtesting.py" --signal-threshold MEDIUM

# Use ROA instead of EPS as the earnings metric
python "Pipeline/run 11 - backtesting.py" --earnings-metric ROA

# Use cached earnings data and skip API calls
python "Pipeline/run 11 - backtesting.py" --skip-fetch
```

Output: `data/backtest/backtest_results.csv`, `data/backtest/regression_summary.txt`

---

## Analyst Tools

These three scripts sit in the project root and provide a standalone interface for scoring individual filings without running the full pipeline.

### Score a single filing or folder

Accepts `.html`, `.pdf`, and `.txt` inputs. Applies both classifiers, computes demeaned scores against section-level baselines, and writes per-document output to `data/analyst_outputs/`.

```powershell
# Single filing
python "RUN USER BACKEND - vague_complex_analysis_pipeline.py" path/to/filing.html

# Folder of filings (mixed formats supported)
python "RUN USER BACKEND - vague_complex_analysis_pipeline.py" path/to/folder/
```

Output: `data/analyst_outputs/{TICKER}_{TYPE}_{DATE}_{ACCESSION}_ddds_scores.csv`

### Desktop GUI

Launches a dark-mode desktop interface wrapping the analyst backend for interactive filing analysis.

```powershell
python "RUN USER FRONTEND - vague_complex_analysis_pipeline.py"
```

### Regenerate the geographic heatmap from analyst outputs

```powershell
python "RUN USER - regional heatmap - analyst outputs.py" \
    --analyst-outputs-dir data/analyst_outputs \
    --save outputs/heatmap.png
```

---

## Full Pipeline Order (Quick Reference)

```
Training (one-time)
    training/run 0     collect training corpus (pre-June 2023 filings)
    training/run 2     GPT-4o consensus labelling — vagueness + complexity
    training/run 3a    generate human annotation sheet
    [manual step]      three annotators label independently
    training/run 3b    merge human labels
    training/run 3     inter-annotator agreement check (kappa >= 0.9 required)
    training/run 4     70/15/15 train/val/test split
    training/run 5     FinBERT fine-tuning — vagueness + complexity

Scoring
    Pipeline/run 0     collect live filings (Oct 2024 - Sep 2025)
    Pipeline/run 0a    clean amendments, enforce filing count per company
    Pipeline/run 1     topic extraction (extract → batch → watch → download)
    Pipeline/run 2     point-in-time peer selection
    Pipeline/run 6     FinBERT inference — vagueness + complexity
    Pipeline/run 7     build Neo4j knowledge graph
    Pipeline/run 8     disclosure screening — GPT-4.1-mini (Tier 1)
    Pipeline/run 9     investment memo generation — Claude Opus (Tier 2)
    Pipeline/run 10    geographic risk heatmap
    Pipeline/run 11    backtesting — Li (2008) earnings persistence regression
```

---

## API Costs

| Component | Estimated cost |
|---|---|
| GPT-4o Batch API labelling — training, 20,000 sentences | ~$4, one-time, cached in repo |
| GPT-4.1-mini screening per quarter (Run 8) | ~$2-5 |
| Claude Opus deep analysis per quarter (Run 9) | ~$10-20 |
| FinBERT inference — Runs 6 and analyst backend | Free (local GPU or CPU) |
| **Full quarterly run** | **~£15-25** |

Cached Batch API results are included in the repository. All downstream outputs can be reproduced from cache at no API cost.

---

## Known Limitations

- The analyst frontend (`RUN USER FRONTEND`) does not currently support full multi-section filing processing. The backend script processes all file types correctly and can be used directly.
- Section boundary detection in Run 1 uses heading pattern matching. Non-standard or heavily nested HTML filings may cause sections to be missed. Check the extraction log for any files reporting zero sentences.
- The training corpus covers 50 randomly seeded companies, which constrains industry diversity within SIC 3400-3599.
- Runs 7 through 11 are not integrated into the analyst frontend and must be run from the command line.