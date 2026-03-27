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
