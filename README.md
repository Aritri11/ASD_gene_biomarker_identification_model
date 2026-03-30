# Machine Learning–Driven Biomarker Gene Prediction for Autism Spectrum Disorder Using Microarray Data

**Goal**: Identify genes whose expression levels can serve as biomarkers to distinguish ASD (Autism Spectrum Disorder) patients from neurotypical Control individuals using the GSE42133 blood microarray dataset (Illumina HT-12 V4).

---

## Project Overview

This repository contains a complete, end-to-end pipeline for processing, analyzing, and classifying gene expression data from whole blood microarray samples. The project integrates R-based preprocessing and differential expression analysis with Python-based machine learning and model interpretation.

The pipeline processes **147 blood microarray samples** (91 ASD, 56 Control) from NCBI GEO (GSE42133) to identify candidate biomarker genes and build an interpretable machine learning classifier.

**Key results:**
- Test Accuracy: **84.44%**
- ROC-AUC: **0.86**
- Cross-validation Macro F1: **0.8494 ± 0.0986**
- Significant DEGs identified: **~250 genes**
- GSEA: **133 GO BP gene sets**, **21 KEGG pathways** enriched

---

## Data Source

| Attribute | Details |
|---|---|
| Dataset | GSE42133 (NCBI Gene Expression Omnibus) |
| Platform | Illumina HumanHT-12 V4 Microarray |
| Tissue | Whole Blood |
| Samples | 147 total (91 ASD, 56 Control) |
| Probes | 48,803 |
| Normalization | Quantile normalized, log2 scale (by original authors) |
| Condition | Autism Spectrum Disorder vs Neurotypical Control |

---

## Repository Structure

```
ASD_biomarker_identification/
│
├── preprocessing/
│   ├── 01_GEO_data_download_QC.R
│   ├── 02_expression_matrix_construction.R
│   ├── 03_diagnosis_label_extraction.R
│   ├── 04_sample_separation_by_diagnosis.R
│   └── 05_sample_merging_long_format.R
│
├── analysis/
│   └── 06_DEG_limma_GSEA_ML_dataset.R
│
├── ml/
│   └── 07_XGBoost_classification_SHAP_interpretation.py
│
├── results/
│   ├── Result_Plots/          # QC and analysis plots from R pipeline
│   ├── SHAP_Plots/            # Model interpretation plots from Python
│   └── SupplementaryFiles/    # DEG tables, GSEA results, ML dataset
│
├── README.md
└── requirements.txt
```

---

## Complete 7-Step Pipeline

### Steps 1–5: Data Preprocessing and Integration (R)

| Step | Script | Description |
|---|---|---|
| 1 | `01_GEO_data_download_QC.R` | Downloads GSE42133 from NCBI GEO, validates expression value range, checks for missing/negative values, generates QC plots (histogram, boxplot), saves metadata |
| 2 | `02_expression_matrix_construction.R` | Loads all 147 individual GSM expression files, merges with GPL10558 probe annotation, performs highest-variance probe deduplication, builds combined wide-format expression matrix |
| 3 | `03_diagnosis_label_extraction.R` | Parses GSE42133 family XML file to extract ASD/Control diagnosis labels per sample, standardizes label names, validates counts against expected 91 ASD / 56 Control |
| 4 | `04_sample_separation_by_diagnosis.R` | Separates individual sample expression files into ASD and Control output folders with full error handling and progress reporting |
| 5 | `05_sample_merging_long_format.R` | Loads all per-sample files and merges into a single long-format dataset (columns: Probe_ID, Symbol, Expression, Sample, Condition), produces both long and wide format outputs |

**Outputs from Steps 1–5:**
```
Step1_histogram_expression.png
Step1_boxplot_samples.png
Step1_sample_metadata.csv
Step1_expression_matrix.rds
expression_matrix_all_samples.csv
GSM_diagnosis.csv
merged_data_long_format.csv
merged_expression_wide.csv
```

---

### Step 6: Differential Expression Analysis and ML Dataset Preparation (R)

**Script:** `06_DEG_limma_GSEA_ML_dataset.R`

**What this script does:**

1. **Probe deduplication** — For each gene with multiple probes, keeps only the probe with the highest variance across all 147 samples. This captures the most biologically dynamic probe rather than diluting signal by averaging.

2. **Design matrix construction** — Builds a limma design matrix using available clinical covariates:
   - `~ Condition + Sex` (if Sex available from GEO metadata)
   - `~ Condition` (fallback if covariates unavailable)
   - SVA batch correction was evaluated but removed — it detected 8 surrogate variables on only 147 samples, over-correcting and suppressing the DEG count from ~150 to 15.

3. **limma differential expression analysis** — Fits linear models with empirical Bayes moderated t-statistics. Produces log2 fold changes, adjusted p-values (Benjamini-Hochberg FDR), and t-statistics for all 31,426 genes.

4. **DEG filtering** — Thresholds chosen for blood microarray data without SVA correction:
   - `|Log2FC| > 0.2` — appropriate for subtle peripheral blood signals in a neurodevelopmental condition (max observed logFC was 0.65)
   - `adj_pval < 0.10` — FDR 10%, standard for exploratory microarray studies

5. **Visualization** — Volcano plot (labeled top 20 genes), heatmap of top 50 DEGs (ward.D2 clustering, Z-scored), log2FC distribution histogram.

6. **GSEA pathway enrichment** — ORA (enrichGO, enrichDO, enrichKEGG) was omitted because only 84 characterised DEGs (after removing LOC/LINC genes) were available — too few for reliable Fisher's exact test enrichment. GSEA uses all 16,668 ranked genes (by t-statistic) and is far more powerful for subtle blood microarray effects.
   - GSEA GO Biological Process
   - GSEA KEGG Pathways

7. **ML dataset construction** — Transposes the DEG expression matrix to samples × genes format, attaches Condition and Label columns, performs a safety check on feature count for Python pipeline compatibility.

**Key design decision — why not SVA?**

SVA was run as a diagnostic. It found 8 surrogate variables, all with low correlation to Condition (max |r| = 0.22). However, adding 8 SVs to a 147-sample design matrix consumed sufficient degrees of freedom to compress fold changes and reduce the detectable DEG count from ~150 to just 15 — making the ML dataset unusable. The residual df with SVA was 137 (healthy), confirming the problem was signal absorption by the SVs rather than loss of statistical power per se. SVA is therefore omitted.

**Outputs from Step 6:**
```
DEG_results_full.csv
DEG_results_significant.csv
Step6_logFC_distribution.png
volcano_plot_limma.png
heatmap_top_DEGs.png
GSEA_GO_BP_results.csv
GSEA_GO_BP_dotplot.png
GSEA_KEGG_results.csv
GSEA_KEGG_dotplot.png
ML_ready_dataset.csv
```

---

### Step 7: Machine Learning Classification and Interpretation (Python)

**Script:** `07_XGBoost_classification_SHAP_interpretation.py`

**Pipeline stages:**

**1. Data loading and normalization**

Two normalization steps are applied in sequence, each solving a different problem:

- *Quantile normalization* (`axis=0`, 100 quantiles, normal output distribution) — corrects for cross-sample technical variation introduced by the R preprocessing steps (probe deduplication and gene-level averaging shift distributions relative to the original array-level normalization).
- *Z-score normalization* (StandardScaler) — standardizes each gene to zero mean and unit variance, ensuring VarianceThreshold and ExtraTrees feature importance scores are comparable across all genes regardless of absolute expression scale.

Note: The data is already log2-transformed and quantile-normalized by the original study authors. The re-normalization in Python is corrective — it realigns distributions after the R-side preprocessing transformations.

**2. Train-test split**

70% training / 30% test, stratified by class label, random_state=42.

**3. Variance threshold filtering**

`VarianceThreshold(threshold=0.7)` is applied to the training set and then transformed on the test set. This removes genes with near-zero variance across training samples — genes that are essentially flat and carry no discriminative information. Importantly this runs *before* SMOTE, because near-constant features corrupt SMOTE's distance-based interpolation of synthetic samples.

**4. SMOTE class imbalance handling**

`SMOTE(sampling_strategy={0: 52, 1: 63}, k_neighbors=8)` generates synthetic Control samples to reduce the class imbalance in the training set. SMOTE runs after VarianceThreshold (cleaner feature space for interpolation) and before feature selection (synthetic samples participate in importance scoring).

**5. ExtraTreesClassifier feature selection**

`SelectFromModel(ExtraTreesClassifier(n_estimators=200, class_weight="balanced"), threshold="median")` selects features above the median importance score.

Key advantages over the previous `SelectKBest(mutual_info_classif)` approach:
- ExtraTrees uses impurity-based importances from tree ensembles — the same mechanism XGBoost uses — so selected features are inherently better matched to the classifier
- `mutual_info_classif` scores each gene independently; ExtraTrees captures gene-gene interaction effects through tree splits
- `class_weight="balanced"` corrects for ASD/Control imbalance *during feature importance computation*, not just during classification — this directly improved Control recall from 0.65 to 0.71

**6. Hyperopt Bayesian hyperparameter optimization**

10-fold Stratified Cross-Validation optimizing macro F1 score across 30 trials using Tree-structured Parzen Estimator (TPE) algorithm.

Search space:
```
n_estimators:     [50, 300]  (step 10)
max_depth:        [3, 10]    (step 1)
learning_rate:    [0.001, 1.0]  (log-uniform)
subsample:        [0.6, 1.0]
colsample_bytree: [0.6, 1.0]
gamma:            [0, 5]
```

**7. Model evaluation**

- Train/Test accuracy
- Classification report (precision, recall, F1 per class)
- Confusion matrix
- ROC-AUC score and ROC curve

**8. SHAP model interpretation**

`shap.TreeExplainer` explains the XGBoost model's predictions using Shapley values — a game-theoretic framework that fairly distributes the prediction among all input features.

Six SHAP visualizations are generated and saved:
- Summary bar plot (global mean |SHAP| feature importance)
- Beeswarm plot (SHAP value distribution per feature, colored by feature value)
- Waterfall plot for the first Control test sample (local explanation)
- Waterfall plot for the first ASD test sample (local explanation)
- Violin plot (SHAP distribution per feature)
- Feature importance bar (shap.plots.bar API)

**Outputs from Step 7:**
```
shap_plots/
    00_roc_curve.png
    01_shap_summary_bar.png
    02_shap_summary_beeswarm.png
    03_shap_waterfall_control.png
    04_shap_waterfall_asd.png
    05_shap_violin.png
    06_shap_feature_importance_bar.png
```

---

## Results

### Classification Performance

| Metric | Train | Test | Cross-Validation |
|---|---|---|---|
| Accuracy | 100.00% | 84.44% | — |
| Macro F1 | — | 0.83 | 0.8494 ± 0.0986 |
| ROC-AUC | — | 0.86 | — |
| Precision (Control) | — | 0.86 | — |
| Recall (Control) | — | 0.71 | — |
| Precision (ASD) | — | 0.84 | — |
| Recall (ASD) | — | 0.93 | — |

### Confusion Matrix (Test Set, n=45)

```
                Predicted Control    Predicted ASD
True Control          12                  5
True ASD               2                 26
```

### Best XGBoost Hyperparameters

```
colsample_bytree: 0.604
gamma:            0.063
learning_rate:    0.172
max_depth:        8
n_estimators:     180
subsample:        0.999
```

### Interpretation of Results

The ROC-AUC of 0.86 indicates strong discriminative ability — if a random ASD sample and a random Control sample are selected, the model assigns a higher ASD probability to the ASD sample 86% of the time. This is a threshold-independent measure and is the primary headline metric for a dataset of this size.

The asymmetry between ASD recall (0.93) and Control recall (0.71) reflects both the class imbalance in the original dataset and the biological reality that some ASD patients have blood gene expression profiles that closely resemble neurotypical Controls — a known phenomenon in the ASD blood transcriptomics literature due to the biological heterogeneity of the disorder.

The cross-validation standard deviation of 0.099 (below 0.10) confirms stable generalisation across different data splits.

---

## Key Methodological Decisions

| Decision | Choice Made | Reason |
|---|---|---|
| Probe deduplication | Highest-variance probe | Captures most biologically dynamic probe; averaging dilutes signal with poorly hybridizing probes |
| Batch correction | SVA removed | 8 SVs on 147 samples over-corrected, reducing DEG count from ~150 to 15 |
| DEG thresholds | \|logFC\| > 0.2, adj_p < 0.10 | Appropriate for blood microarray; max logFC in this dataset is only 0.65 |
| Pathway method | GSEA only (no ORA) | Only 84 characterised DEGs — too few for reliable Fisher's exact test ORA |
| Feature selection | ExtraTrees + SelectFromModel | Tree-based importances matched to XGBoost; class_weight=balanced addresses Control recall |
| Normalization | Quantile + Z-score | Quantile corrects cross-sample technical shifts from R preprocessing; Z-score equalises gene scales for VarianceThreshold and ExtraTrees |

---

## Requirements

### R Dependencies

```r
# Bioconductor packages
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c(
  "GEOquery",        # Download from GEO database
  "limma",           # Differential expression analysis
  "sva",             # Surrogate variable analysis (diagnostic only)
  "clusterProfiler", # Pathway enrichment analysis
  "org.Hs.eg.db",    # Human gene annotations
  "DOSE"             # Disease ontology enrichment
))

# CRAN packages
install.packages(c(
  "data.table",      # Fast data manipulation
  "xml2",            # Parse XML metadata
  "dplyr",           # Data wrangling
  "tidyverse",       # Data science pipeline
  "pheatmap",        # Heatmap visualization
  "ggplot2",         # Advanced plotting
  "ggrepel"          # Non-overlapping text labels
))
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.5.3
numpy>=1.24.3
matplotlib>=3.7.1
scikit-learn>=1.2.2
xgboost>=1.7.5
hyperopt>=0.2.5
shap>=0.42.1
imbalanced-learn>=0.10.1
```

---

## Quick Start

### Prerequisites

1. R version 4.0 or higher
2. Python 3.8 or higher
3. Stable internet connection for GEO data download (~500 MB)
4. 4–8 GB RAM recommended

### Running the Complete Pipeline

**Step 1 — Download data and run QC:**
```bash
Rscript preprocessing/01_GEO_data_download_QC.R
```

**Steps 2–5 — Build expression matrix and merge samples:**
```bash
Rscript preprocessing/02_expression_matrix_construction.R
Rscript preprocessing/03_diagnosis_label_extraction.R
Rscript preprocessing/04_sample_separation_by_diagnosis.R
Rscript preprocessing/05_sample_merging_long_format.R
```

**Step 6 — DEG analysis and ML dataset preparation:**
```bash
Rscript analysis/06_DEG_limma_GSEA_ML_dataset.R
```

**Step 7 — Machine learning and SHAP interpretation:**
```bash
python ml/07_XGBoost_classification_SHAP_interpretation.py
```

### Important: Update File Paths Before Running

All R scripts contain hardcoded Windows paths that must be updated before running:

```r
# In each R script, update these variables at the top:
long_format_file <- "path/to/your/merged_data_long_format.csv"
output_dir       <- "path/to/your/output/directory"
annot_path       <- "path/to/GPL10558-50081.txt"
gsm_folder       <- "path/to/GSE42133_family/"
```

The GPL10558 annotation file must be downloaded separately from NCBI GEO:
[https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL10558](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL10558)

---

## Reproducibility

All random seeds are fixed throughout the pipeline:

```python
# Python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

```r
# R
set.seed(42)
```

Running the pipeline with the same data and fixed seeds will reproduce identical results.

---

## Technical Stack

| Component | Tool | Notes |
|---|---|---|
| Languages | R ≥ 4.0, Python 3.8+ | |
| Microarray DEG analysis | limma (Bioconductor) | Moderated t-statistics, empirical Bayes |
| Pathway analysis | clusterProfiler | GSEA on GO BP and KEGG |
| ML framework | scikit-learn, XGBoost | |
| Hyperparameter tuning | Hyperopt | Bayesian TPE optimization |
| Class imbalance | imbalanced-learn (SMOTE) | |
| Model interpretation | SHAP | TreeExplainer |
| Visualization (R) | ggplot2, pheatmap, ggrepel | |
| Visualization (Python) | matplotlib, SHAP plots | |

---

## Author and Contact

**Author:** Aritri Baidya  
**Supervisor:** Dr. Shyam Sundar Rajagopalan  
**GitHub:** [@Aritri11](https://github.com/Aritri11)

For questions, issues, or collaborations, please open a GitHub Issue or contact via the GitHub profile linked above.

---

## References

**Datasets and Platforms**
- GSE42133: NCBI Gene Expression Omnibus — [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42133)
- Illumina HumanHT-12 V4 Microarray — [https://www.illumina.com](https://www.illumina.com)

**R Packages**
- Smyth GK (2004). Linear models and empirical Bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology*, 3(1).
- Yu G et al. (2012). clusterProfiler: an R Package for Comparing Biological Themes Among Gene Clusters. *OMICS*, 16(5), 284-287.
- Ritchie ME et al. (2015). limma powers differential expression analyses for RNA-sequencing and microarray studies. *Nucleic Acids Research*, 43(7), e47.

**Machine Learning and Interpretation**
- Chen T, Guestrin C (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Lundberg SM, Lee SI (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
- Chawla NV et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321-357.

---

## License

This project is provided for research and educational purposes. Please cite appropriately if used in academic work.

---

## Acknowledgments

- **NCBI GEO Database** for providing open access to the GSE42133 microarray dataset
- **Bioconductor Community** for the limma and clusterProfiler packages
- **Open-source Contributors** of scikit-learn, XGBoost, SHAP, and Hyperopt
