# Claims Comparison & Validation Tool (Old vs New Processing)

This repo contains a **small, reliable data comparison tool** to validate that a replacement healthcare claims processing system produces outputs consistent with an incumbent system.

It is designed for a **parallel run / cutover** scenario:

- **Old system outputs** are treated as the baseline (ground truth).
- **New system outputs** are compared against the baseline.
- The tool **surfaces issues**, **quantifies impact**, highlights **trends**, and produces an **Excel report** for review.

## Why Python + SQLite?

- **Python**: fast iteration, great ecosystem for data QA, easy to explain.
- **SQLite**: embedded DB (no server), supports SQL joins/aggregations for large files; keeps the pipeline reproducible and portable.

If you want to scale further, the comparison logic can be swapped to DuckDB/Postgres without changing the report layer.

---

## Inputs

### Required (baseline / old system)
Download from the CMS DE-SynPUF Sample 1 page:

- 2008 Beneficiary Summary (ZIP)
- 2009 Beneficiary Summary (ZIP)
- 2010 Beneficiary Summary (ZIP)
- 2008–2010 Carrier Claims 1 (ZIP)
- 2008–2010 Carrier Claims 2 (ZIP)

CMS download page: citeturn1view0

> Note: In this ChatGPT sandbox, we were able to download the beneficiary summary ZIPs directly from `cms.gov`. Carrier claim ZIPs may require downloading outside the sandbox and placing them in `data/old/`.

### Required (candidate / new system)
A ZIP containing “comparable outputs” from the replacement system (provided in the prompt).

Unzip into `data/new/`, keeping filenames aligned with the baseline.

---

## Quick Start

```bash
# 1) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Place baseline outputs in data/old/ and new outputs in data/new/

# 4) Run the comparison
python src/compare.py --old data/old --new data/new --out reports --db reports/staging.sqlite
```

Outputs:

- `reports/claims_comparison_report.xlsx` (primary deliverable)
- `reports/staging.sqlite` (staged tables + joins; useful for debugging)

---

## What the Excel report includes

- **Run_Metadata**: inputs, tolerance, run timestamp.
- **Overview**: roll-up counts (rows, missing keys, duplicates).
- **Key_Issues**: missing/extra/duplicate key counts per table.
- **Payment_Field_Mismatch**: mismatch rates and $ impact per payment field.
- **Top_Differences**: top 50 absolute diffs per field (drill-down).
- **Trend_* tabs**: mismatches by state and race to spot systemic issues.
- **Sample_* tabs**: example missing keys, duplicated keys, etc.

---

## Extending to Claim-level Comparisons

This submission demonstrates beneficiary-year level payment totals (`MEDREIMB_*`, `BENRES_*`, `PPPYMT_*`).

To extend to claim-level checks (Carrier Claims):

1. Load Carrier Claim CSVs into SQLite.
2. Choose business keys (e.g., `DESYNPUF_ID + CLM_ID`).
3. Compare monetary outputs (e.g., `LINE_NCH_PMT_AMT`, allowed/charge amounts).
4. Add:
   - trends by provider specialty, place-of-service, HCPCS code
   - distribution drift checks (percentiles, tails)

The same mismatch engine can be reused (missing/extra/duplicate/value mismatch).

---

## Repo Layout

```
.
├── src/
│   └── compare.py          # CLI comparison tool
├── data/
│   ├── old/                # baseline outputs
│   └── new/                # candidate outputs
└── reports/
    ├── claims_comparison_report.xlsx
    └── staging.sqlite
```

---

## Notes on CMS DE-SynPUF

CMS describes Sample 1 as part of the 2008–2010 Data Entrepreneurs’ Synthetic PUF and provides separate beneficiary-year files and multi-year claims files (carrier split into 2 parts due to size). citeturn1view0
