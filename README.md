# Poverty Is Falling Everywhere: Convergence, Not Dynasty, in Philippine Local Government

## 📖 Project Overview
This repository contains the data pipeline, econometric scripts, and visualization code for the research paper: **"Poverty Is Falling Everywhere: Convergence, Not Dynasty, in Philippine Local Government."**

The research re-evaluates the widely held hypothesis that political dynasties in the Philippines slow down poverty reduction. Using a municipal-level panel of 1,714 Local Government Units (LGUs) across three waves (2018, 2021, 2023), the study finds that the observed positive association between dynasties and poverty is largely a statistical artifact of **mean reversion** (convergence) and **clustering methodology**, rather than a dynasty-specific drag on development.

## 🔑 Key Findings
*   **The Convergence Artifact:** Raw data shows dynastic LGUs declining in poverty *faster* than non-dynastic LGUs. However, this is because they started 3–4 percentage points poorer in 2018. Once initial poverty is controlled, the dynasty effect collapses to zero.
*   **Clustering Fragility:** The continuous measure of dynasty exposure (share of dynastic officials) loses statistical significance when standard errors are properly clustered at the provincial level, rather than the LGU level.
*   **Bounded Null for "Fat" Dynasties:** The theoretically central distinction between "fat" dynasties (Mayor and Vice Mayor are same-clan) and "thin" dynasties is statistically indistinguishable from zero (`+1.17 pp`, 95% CI `[-1.26, 3.60]`).
*   **No Selection Bias:** LGUs that would go on to become dynastic ("Entry") were not statistically poorer in 2018 than LGUs that remained non-dynastic ("Never").

---

## 📊 Visualizing the Results

### 1. Clustering Sensitivity
The continuous share measure crosses the zero line under province clustering, while the binary measure remains robust.
![Clustering Sensitivity](output/paper2/figures/fig1_clustering_sensitivity.png)

### 2. Poverty Trajectories by Dynasty Entry Group
All groups experience poverty decline. The dynastic groups start significantly poorer but exhibit steeper declines.
![Poverty Trajectories](output/paper2/figures/fig2_trajectories.png)

### 3. Convergence in Poverty Reduction (2018–2023)
Initial poverty strongly predicts subsequent decline. Dynastic and non-dynastic LGUs follow nearly identical convergence paths.
![Convergence Scatter](output/paper2/figures/figA_convergence_scatter.png)

### 4. Fat vs. Thin Dynasties: A Bounded Null
The difference between same-clan and other-dynastic LGUs has a confidence interval that tightly straddles zero.
![Fat vs Thin](output/paper2/figures/fig4_fat_vs_thin.png)

### 5. Pre-Transition Null: Entry vs. Never
Pre-transition poverty levels between entrants and non-dynastic LGUs are statistically identical, mitigating selection concerns.
![Entry vs Never](output/paper2/figures/figB_entry_vs_never.png)

### 6. Heterogeneity in the Dynasty-Poverty Association
The association is concentrated in 2018 and disappears by 2023. It is driven primarily by Mindanao.
![Heterogeneity](output/paper2/figures/fig3_heterogeneity.png)

---

## 📂 Repository Structure
```text
dynasty-poverty/
│
├── data/
│   └── processed/
│       ├── panel_dynasty_funds.parquet     # Master municipal panel dataset
│       ├── openhalalan_candidates.parquet  # Candidate dynastic classifications
│       ├── psgc_lgu_master.parquet         # LGU geographic reference
│       └── psgc_provinces_master.parquet   # Province geographic reference
│
├── src/
│   ├── audit_reviewers.py                  # Core econometric engine (TWFE models, clustering, diagnostics)
│   ├── generate_figure_data.py             # Exports plotting data (CSV) from the audit models
│   └── plot_figures.py                     # Renders publication-ready PNGs from the CSV data
│
├── output/
│   └── paper2/
│       ├── audit_reviewers.txt             # Full regression output and diagnostics log
│       ├── audit_reviewers.csv             # Tidy format of all regression coefficients
│       └── figures/                        # Final figures for the manuscript (PNG) & raw data (CSV)
│
├── README.md
└── requirements.txt
```

## 🛠️ Methodology & Pipeline

The analysis uses **Two-Way Fixed Effects (TWFE) OLS regression** with standard errors clustered at the provincial level. The pipeline is split into three distinct phases to ensure reproducibility and allow for tweaking of figures without rerunning the expensive regressions.

### 1. Data & Audit (`src/audit_reviewers.py`)
This is the core econometric script. It performs the following:
*   Constructs the `psgc10` keys (Region + Province + Municipality).
*   Builds the "Fat vs. Thin" dynasty classification.
*   Runs TWFE models with standard errors clustered at both the **LGU** and **Province** levels.
*   Runs the long-difference convergence models.
*   Outputs `audit_reviewers.txt` and `.csv` to `output/paper2/`.

### 2. Figure Data Generation (`src/generate_figure_data.py`)
This script pulls from the audit models and the main panel to generate exactly the data needed for the plots.
*   Calculates binned scatter data for the convergence plots.
*   Extracts coefficients and confidence intervals into a clean, plotting-friendly CSV format.
*   Outputs `.csv` files to `output/paper2/figures/`.

### 3. Visualizations (`src/plot_figures.py`)
This script uses `matplotlib` and `seaborn` to render the final 300 DPI publication-ready images.
*   Plots the Clustering Sensitivity, Trajectories, Heterogeneity, and Bounded Nulls.
*   Outputs `.png` files to `output/paper2/figures/`.

---

## 🚀 How to Reproduce

### Prerequisites
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install pandas numpy statsmodels scipy matplotlib seaborn pyarrow
```

### Execution
Run the scripts in the following order from the project root:

**Step 1: Run the main econometrics audit**
```bash
python -u src/audit_reviewers.py
```
*Check `output/paper2/audit_reviewers.txt` to verify the regression results.*

**Step 2: Generate the figure datasets**
```bash
python -u src/generate_figure_data.py
```
*Check `output/paper2/figures/` for the generated CSV files.*

**Step 3: Render the final figures**
```bash
python -u src/plot_figures.py
```
*The final `.png` figures for the manuscript will now be in `output/paper2/figures/`.*

---

## 📝 Citation
If you use this code or data, please cite the forthcoming manuscript:
> [Author Names]. (2026). Poverty Is Falling Everywhere: Convergence, Not Dynasty, in Philippine Local Government. *Poverty and Public Policy*.

## ⚠️ Notes on Data Availability
The `.parquet` files in `data/processed/` are derived from public records (COMELEC, PSA, BLGF). Ensure you have the appropriate data use agreements if sharing these `.parquet` files publicly.
```

### One quick tip for GitHub:
Once you push this to GitHub, make sure that the `output/paper2/figures/` folder and the `.png` files inside it are **actually committed to the repository**. If your `.gitignore` is set to ignore images or the `output` folder, the images won't show up on the GitHub README! 

To force-add them if they are being ignored, you can run:
```powershell
git add -f output/paper2/figures/*.png
git commit -m "Add figures for README display"
git push
```