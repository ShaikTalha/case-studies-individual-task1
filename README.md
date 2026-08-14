# Case Studies in Data Science — Individual Task 1

ML analysis code for RMIT "Case Studies in Data Science", Individual Task 1, Part 1.3.

**Author:** Talha Shaik (S4150701), RMIT University
**Report:** [Overleaf project](https://www.overleaf.com/project/6a7dd24d6868edeba0e3b880) (see Section 4 and Appendix A)

## Overview

This repository contains the machine learning analysis referenced in the report. It trains and
evaluates two algorithms — a Decision Tree and a Neural Network (MLP) — on each of two public
healthcare datasets, and generates the figures and metrics used in Section 4 (Data Analysis).

## Contents

| File / folder | Description |
|---|---|
| `analysis.py` | Main analysis script. Run with `python analysis.py`. |
| `analysis.ipynb` | Google Colab-ready notebook version of the same analysis, with outputs already run. |
| `results.json` | Output metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrices) for both datasets and both algorithms. |
| `figures/` | Generated plots (confusion matrices, ROC curves, Decision Tree feature importance) used in the report. |
| `data/hac_infection_2023.csv` | Dataset B, included directly (see below). |
| `data/diabetic_data.csv` | Dataset A — **not included** in this repo (~19 MB, above a convenient GitHub web-upload size). See download instructions below. |

## Datasets

**Dataset A (patient-level):** UCI *"Diabetes 130-US hospitals for years 1999-2008"*
Canonical source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
Mirror used to obtain the CSV: https://github.com/andrewwlong/diabetes_readmission/blob/master/diabetic_data.csv
Target: hospital readmission within 30 days (binary).

To reproduce locally: download `diabetic_data.csv` from the mirror above (click **Raw** → **Download raw file**) and place it in `data/`.

**Dataset B (facility-level):** CMS Hospital-Acquired Condition (HAC) Reduction Program infection data, 2023
Source: https://github.com/klocey/hospitals-data-archive (curated from CMS Care Compare / data.cms.gov)
Target: elevated healthcare-associated infection risk (mean SIR > 1.0, binary).
Already included at `data/hac_infection_2023.csv`.

## How to run

**Locally:**
```bash
pip install scikit-learn pandas matplotlib numpy
# place diabetic_data.csv in data/ (see above) — hac_infection_2023.csv is already there
python analysis.py
```
Outputs are written to `results.json` and `figures/`.

**Google Colab:**
Open `analysis.ipynb` in Colab and run all cells. If the data files aren't found, the notebook
will prompt a file-upload widget so you can select them from your computer.

## Results summary

| Dataset | Algorithm | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| A — Diabetes readmission | Decision Tree | 0.603 | 0.115 | 0.512 | 0.188 | 0.585 |
| A — Diabetes readmission | Neural Network | 0.601 | 0.107 | 0.469 | 0.174 | 0.560 |
| B — Hospital infection risk | Decision Tree | 0.547 | 0.367 | 0.514 | 0.429 | 0.566 |
| B — Hospital infection risk | Neural Network | 0.351 | 0.333 | 0.964 | 0.495 | 0.472 |

Full metrics (including confusion matrices and top predictive features) are in `results.json`.
Discussion and interpretation are in Section 4 of the report.

## Job listing & cover letter

The job listing this analysis is grounded in, and the accompanying cover letter, are reproduced
in Appendices A and B of the report (Overleaf link above) — not duplicated here.
