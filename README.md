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
├── Pipeline/                        # Scoring pipeline (Runs 0-8)
│   ├── run_00_sec_data_collection.py
│   ├── run_00a_clean_amendments.py
│   ├── run_01_topic_extraction.py
│   ├── run_02_peer_selection.py
│   ├── run_03_vague_predictions.py
│   ├── run_03_complex_predictions.py
│   ├── run_04_graph_building.py
│   ├── run_05_topic_screening.py
│   ├── run_06_opus_analysis.py
│   ├── run_06_memo_generation.py
│   ├── run_07_risk_heatmap.py
│   ├── run_07_opus_memo_generation.py
│   ├── run_08_backtesting.py
│   ├── cache.py
│   └── resumebatches.py
│
├── training/                        # Training pipeline (one-time)
│   ├── run_00_sec_data_collection.py
│   ├── run_02_vague_labelling.py
│   ├── run_02_complex_labelling.py
│   ├── run_03_human_agreement_checker.py
│   ├── run_03a_generate_validation_sheet.py
│   ├── run_03b_merge_human_labels.py
│   ├── run_04_ttv_split.py
│   ├── run_05_train_vagueness.py
│   └── run_05_train_complexity.py
│
├── scrapers/                        # Standalone data collection scripts
│   ├── 8k_scraper.py
│   ├── edgar_api.py
│   └── sec_api_scraper.py
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
├── analyst_backend.py
├── analyst_frontend.py
├── analyst_heatmap.py
├── requirements.txt
├── .env                             # Not committed — see Setup below
└── README.md
```

---

## Prerequisites

- Python 3.11
- An OpenAI API key with Batch API access
- An Anthropic API key
- The Neo4j AuraDB password (provided in the accompanying technical summary — no local Neo4j installation required)

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
NEO4J_PASSWORD=see-technical-summary
NEO4J_URI=see-technical-summary
NEO4J_USER=see-technical-summary
EDGAR_USER_AGENT= your-email@emailprovider.com
FINDINGS_DIR=data/findings
ANALYST_OUTPUTS_DIR=data/analyst_outputs
SEC_USER_AGENT= email@emailprovider.com

```

The pipeline connects to a hosted **Neo4j AuraDB** (free tier) instance containing the full pre-populated graph. The connection URI and username are pre-configured in the pipeline scripts — only `NEO4J_PASSWORD` needs to be set. The password is provided in the accompanying technical summary. No local Neo4j installation is required.

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
python "training/run_00_sec_data_collection.py"
```

Output: `data/training_sentences/training_sentences.csv`

---

### Training Run 2 — GPT-4o Consensus Labelling

Submits two independent Batch API jobs per classifier. Only sentences where both runs agree are carried forward. Run both vagueness and complexity labelling.

```powershell
python "training/run_02_vague_labelling.py"
python "training/run_02_complex_labelling.py"
```

Batch jobs are asynchronous. Monitor progress with:

```powershell
python "training/run_02_vague_labelling.py" --mode status
```

If a batch was interrupted, resume from where it left off:

```powershell
python "Pipeline/resumebatches.py"
```

To replay from the included cache without re-submitting:

```powershell
python "training/run_02_vague_labelling.py" --use-cached
python "training/run_02_complex_labelling.py" --use-cached
```

Output: `data/labelled/vagueness/labelled.csv`, `data/labelled/complexity/labelled.csv`

---

### Training Run 3a — Generate Validation Sheet

Stratifies 300 consensus sentences per classifier (150 per class) into a formatted Excel annotation sheet for human review. The sheet is split across three annotators at 100 sentences each.

```powershell
python "training/run_03a_generate_validation_sheet.py"
```

Output: `data/validation/validation_annotation_sheet.xlsx`

**Manual step:** each annotator labels their 100 sentences independently and returns the completed sheet. The three completed sheets are then merged before proceeding to Run 3b.

---

### Training Run 3b — Merge Human Labels

Merges the three completed annotation sheets into a single labelled CSV and applies the `human_validated` flag used in downstream stratification.

```powershell
python "training/run_03b_merge_human_labels.py"
```

---

### Training Run 3 — Inter-Annotator Agreement Check

Computes Cohen's Kappa for all three pairwise annotator combinations and for GPT-4o against the human majority vote. A Kappa of at least 0.9 across all pairings is required before proceeding.

```powershell
python "training/run_03_human_agreement_checker.py"
```

---

### Training Run 4 — Train/Val/Test Split

Constructs a 3,000-sentence dataset per classifier. All 300 human-validated rows are guaranteed to be included. The remaining 2,700 are stratified-sampled from the GPT-labelled pool. Applies a 70/15/15 split stratified jointly on label and `human_validated`.

```powershell
python "training/run_04_ttv_split.py"
```

Output: `data/training/vagueness/train.csv`, `val.csv`, `test.csv`  
Output: `data/training/complexity/train.csv`, `val.csv`, `test.csv`

---

### Training Run 5 — Fine-Tune FinBERT

Fine-tunes two binary classifiers independently from the finbert checkpoint. Training uses weighted cross-entropy for class imbalance, macro F1 as the best-model selection criterion, and early stopping with patience of 2.

```powershell
python "training/run_05_train_vagueness.py"
python "training/run_05_train_complexity.py"
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
python "Pipeline/run_00_sec_data_collection.py"
```

On subsequent runs, reuse the cached universe to skip the ~19-minute rebuild:

```powershell
python "Pipeline/run_00_sec_data_collection.py" --use-cached
```

Output: `data/Filings by company/{TICKER}/10-K/` and `10-Q/`  
Output: `company_filing_summary.xlsx`

---

### Run 0a — Clean Amendments

Removes any amendment files (10-K-A, 10-Q-A) and enforces that every company folder contains exactly 2 x 10-K and 3 x 10-Q. Companies that fail this check are removed from the universe.

```powershell
python "Pipeline/run_00a_clean_amendments.py" --data-dir "data/Filings by company"
```

To preview changes without deleting anything:

```powershell
python "Pipeline/run_00a_clean_amendments.py" --data-dir "data/Filings by company" --dry-run
```

---

### Run 1 — Topic Extraction

Extracts Risk Factors and MD&A sections from each HTML filing, splits them into sentences, and classifies each sentence into one of 12 topic codes using GPT-4o-mini via the Batch API.

Run in sequence:

```powershell
# Step 1: Extract sentences from all filings
python "Pipeline/run_01_topic_extraction.py" --mode extract

# Step 2: Submit Batch API job for topic labelling
python "Pipeline/run_01_topic_extraction.py" --mode batch

# Step 3: Monitor until complete (auto-polls every 60 seconds)
python "Pipeline/run_01_topic_extraction.py" --mode watch

# Step 4: Download results and merge into final CSV
python "Pipeline/run_01_topic_extraction.py" --mode download
```

To use the included cached results and skip the batch submission entirely:

```powershell
python "Pipeline/run_01_topic_extraction.py" --mode download --use-cached
```

Output: `data/Graph Rag Creation Data/topic_labelled.csv`

---

### Run 2 — Peer Selection

Constructs a point-in-time peer group for each company filing using Euclidean distance across three normalised financial metrics: market capitalisation, trailing twelve-month revenue, and total assets. Peer candidates are identified via tiered SIC grouping (4-digit first, falling back to 3-digit then 2-digit). The five closest peers are written to CSV and later ingested into the Neo4j graph as `PEER_OF` edges.

```powershell
python "Pipeline/run_02_peer_selection.py" --input-dir "data/Filings by company"
```

On subsequent runs, reuse cached financial data from yfinance:

```powershell
python "Pipeline/run_02_peer_selection.py" --use-cached
```

Output: `data/Graph Rag Creation Data/peer_selections.csv`

---

### Run 3 — FinBERT Inference

Runs batch inference with the fine-tuned classifiers on all sentences in `topic_labelled.csv`. Both classifiers must be run.

```powershell
python "Pipeline/run_03_vague_predictions.py" \
    --model models/finbert_vagueness \
    --input "data/Graph Rag Creation Data/topic_labelled.csv" \
    --output data/predictions_vagueness.csv

python "Pipeline/run_03_complex_predictions.py" \
    --model models/finbert_complexity \
    --input "data/Graph Rag Creation Data/topic_labelled.csv" \
    --output data/predictions_complexity.csv
```

Output: `data/predictions_vagueness.csv`, `data/predictions_complexity.csv`

---

### Run 4 — Build Neo4j Graph

Ingests `topic_labelled.csv` and `peer_selections.csv` into the hosted Neo4j AuraDB knowledge graph. The connection is pre-configured — ensure `NEO4J_PASSWORD` is set in your `.env` before running.

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
python "Pipeline/run_04_graph_building.py"
```

To wipe the database and rebuild from scratch:

```powershell
python "Pipeline/run_04_graph_building.py" --wipe
```

---

### Run 5 — Disclosure Screening (Graph-RAG Tier 1)

Queries the Neo4j graph for disclosure block pairs and screens them for degradation using GPT-4.1-mini. Two comparison layers are applied: temporal (same company, current vs prior period) and peer (same topic, current company vs each of its five peers). Disappeared topics are detected via graph traversal without any LLM call. Flagged results are written to CSV for escalation in Run 6.

```powershell
python "Pipeline/run_05_topic_screening.py"
```

To run a quick test on five random temporal pairs before the full run:

```powershell
python "Pipeline/run_05_topic_screening.py" --test
```

To use the LLM response cache and avoid repeated API calls on re-runs:

```powershell
python "Pipeline/run_05_topic_screening.py" --use-cached
```

Output: `data/Graph Rag Creation Data/screening_flags.csv`

---

### Run 6 — Investment Memo Generation (Graph-RAG Tier 2)

Passes all flagged items per company to Claude Opus for deep analysis covering temporal deterioration, peer comparison, and cross-document contradictions. Generates a formatted two-page `.docx` investment memo per flagged company. All outputs include model provenance and confidence levels.

```powershell
python "Pipeline/run_06_memo_generation.py"
```

Output: `data/memos/{TICKER}_memo.docx`

---

### Run 8 — Geographic Risk Heatmap

Produces a global heatmap attributing raw vagueness scores to countries based on geographic keyword matching within each scored sentence. The USA receives scores from all sentences across the universe. No demeaning is applied to this output.

```powershell
python "Pipeline/run_08_risk_heatmap.py"
```

To export a PNG without opening the interactive window:

```powershell
python "Pipeline/run_08_risk_heatmap.py" --save outputs/heatmap.png --headless
```

Output: `outputs/heatmap.png`

---

### Run 9 — Backtesting

Tests the core hypothesis using the Li (2008) earnings persistence regression. Flagged companies are expected to show lower earnings persistence in the period following the flag relative to unflagged peers. The regression is run across three subsamples: full sample, profitable companies only (earnings > 0), and loss-making companies only (earnings < 0). Earnings data is pulled from the SEC EDGAR XBRL API with a yfinance fallback.

```powershell
python "Pipeline/run_09_backtesting.py"
```

Optional arguments:

```powershell
# Set minimum signal level to count as flagged (HIGH, MEDIUM, or LOW)
python "Pipeline/run_08_backtesting.py" --signal-threshold MEDIUM

# Use ROA instead of EPS as the earnings metric
python "Pipeline/run_08_backtesting.py" --earnings-metric ROA

# Use cached earnings data and skip API calls
python "Pipeline/run_08_backtesting.py" --skip-fetch
```

Output: `data/backtest/backtest_results.csv`, `data/backtest/regression_summary.txt`

---

## Analyst Tools

These three scripts sit in the project root and provide a standalone interface for scoring individual filings without running the full pipeline.

### Score a single filing or folder

Accepts `.html`, `.pdf`, and `.txt` inputs. Applies both classifiers, computes demeaned scores against section-level baselines, and writes per-document output to `data/analyst_outputs/`.

```powershell
# Single filing
python analyst_backend.py path/to/filing.html

# Folder of filings (mixed formats supported)
python analyst_backend.py path/to/folder/
```

Output: `data/analyst_outputs/{TICKER}_{TYPE}_{DATE}_{ACCESSION}_ddds_scores.csv`

### Desktop GUI

Launches a unified dark-mode desktop application built with CustomTkinter and Matplotlib. Select any ticker from the sidebar to navigate its analysis across five tabs:

| Tab | What it shows |
|---|---|
| **Investment Memo** | Full Opus-generated analysis for the selected company: signal strength badge, executive summary, genuine / borderline / dismissed flag counts, top concerns, and per-flag reasoning cards with investigation actions. |
| **Risk Heatmap** | Sector-wide geographic vagueness map. Countries are coloured by mean raw `vague_prob` of all sentences that mention them across the 925-filing universe. Hover over any country to see its score and sentence-count detail. This view is independent of the selected ticker. |
| **Filing Trends** | Per-company time-series chart showing mean `vague_prob`, `complex_prob`, and their average (`avg_score`) across every scored filing for the selected ticker. Circle markers = 10-K, triangle markers = 10-Q. |
| **Sector Trends** | Sector-wide quarterly averages of all three scores, aggregated across all 925 filings in 3-month buckets. Each data point shows the mean ± 1 standard deviation band across all companies that filed in that quarter, plus a filing count annotation (`n=`). Loaded in the background at startup. |
| **FinBERT Rankings** | Top 15 companies by most extreme FinBERT scores, drawn from each company's most recent filing only. Only filings with at least 50 sentences across MD&A and Risk Factors combined are eligible. Toggle the ranking metric between **Vague**, **Complex**, and **Both** (average). The right panel has three views toggled independently: **Stock Price** (% return from filing date to today via yfinance), **Earnings Persistence** (SOE at filing date vs most recent SOE, with persistence rated HIGH / MODERATE / LOW based on the magnitude of change normalised by the company's own pre-filing earnings standard deviation), and **Stats** (vague%, sentence count, filing date). |

```powershell
python analyst_frontend.py
```

### Regenerate the geographic heatmap from analyst outputs

```powershell
python analyst_heatmap.py \
    --analyst-outputs-dir data/analyst_outputs \
    --save outputs/heatmap.png
```

---

## Full Pipeline Order (Quick Reference)

```
Training (one-time)
    training/run_00    collect training corpus (pre-June 2023 filings)
    training/run_02    GPT-4o consensus labelling — vagueness + complexity
    training/run_03a   generate human annotation sheet
    [manual step]      three annotators label independently
    training/run_03b   merge human labels
    training/run_03    inter-annotator agreement check (kappa >= 0.9 required)
    training/run_04    70/15/15 train/val/test split
    training/run_05    FinBERT fine-tuning — vagueness + complexity

Scoring
    Pipeline/run_00    collect live filings (Oct 2024 - Sep 2025)
    Pipeline/run_00a   clean amendments, enforce filing count per company
    Pipeline/run_01    topic extraction (extract → batch → watch → download)
    Pipeline/run_02    point-in-time peer selection
    Pipeline/run_03    FinBERT inference — vagueness + complexity
    Pipeline/run_04    build Neo4j knowledge graph
    Pipeline/run_05    disclosure screening — GPT-4.1-mini (Tier 1)
    Pipeline/run_06    Opus analysis + investment memo generation — Claude Opus (Tier 2)
    Pipeline/run_07    geographic risk heatmap + memo generation
    Pipeline/run_08    backtesting — Li (2008) earnings persistence regression
```

---

## API Costs

| Component | Estimated cost |
|---|---|
| GPT-4o Batch API labelling — training, 20,000 sentences | ~$4, one-time, cached in repo |
| GPT-4.1-mini screening per quarter (Run 5) | ~$2-5 |
| Claude Opus deep analysis per quarter (Run 6) | ~$10-20 |
| FinBERT inference — Run 3 and analyst backend | Free (local GPU or CPU) |
| **Full quarterly run** | **~£15-25** |

Cached Batch API results are included in the repository. All downstream outputs can be reproduced from cache at no API cost.

---

## Known Limitations

- The **Investment Memo** tab displays pre-computed Opus analysis for the 128-company universe. The **Filing Trends**, **Sector Trends**, **Risk Heatmap**, and **FinBERT Rankings** tabs all derive directly from the 925 scored CSVs in `data/analyst_outputs/` and will reflect any new filings scored via `analyst_backend.py`. Scoring companies outside the existing universe still requires running the full pipeline from the command line to generate the Opus analysis JSON.
- The **FinBERT Rankings** earnings persistence calculation requires quarterly earnings data in `data/backtest/quarterly_earnings_yfinance.csv` (produced by Run 11). Companies not present in that file will show `N/A` for persistence. Stock price data is fetched live from yfinance at startup; tickers that are delisted or not covered will show `n/a` for price return.
- Section boundary detection in Run 1 uses heading pattern matching. Non-standard or heavily nested HTML filings may cause sections to be missed. Check the extraction log for any files reporting zero sentences.
- The training corpus covers 50 randomly seeded companies, which constrains industry diversity within SIC 3400-3599.
- Runs 4 through 8 must be run from the command line; they are not triggered from within the frontend.
