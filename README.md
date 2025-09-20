Machine Learning–Driven Biomarker Gene Prediction for Autism Spectrum Disorder Using RNA-Seq Data

Goal: Identify genes whose expression levels can serve as biomarkers to distinguish ASD (Autism Spectrum Disorder) patients from Control individuals.

This repository contains a complete pipeline for processing, analyzing, and classifying gene expression data from the GSE42133 dataset.
The project integrates R-based preprocessing & differential expression analysis with Python-based machine learning and SHAP interpretation to classify Autism Spectrum Disorder (ASD) vs Control samples and Identify genes whose expression levels can serve as biomarkers to distinguish ASD (Autism Spectrum Disorder) patients from Control individuals.

📌 Workflow Overview

1. Data Retrieval & Preprocessing (R)

  * Download GEO dataset (GSE42133) using GEOquery.

  * Extract expression values and phenotype metadata.
  
  * Merge probe-level data with gene annotations (GPL10558).
  
  * Parse XML metadata to map samples (GSM IDs) to diagnoses (ASD/Control).
  
  * Split raw expression data into ASD and Control subsets.

2. Data Integration

  * Merge sample-wise expression files into a long-format dataset.
  
  * Transform into gene × sample matrix.
  
  * Annotate with condition labels (ASD/Control).

3. Differential Expression Analysis (R)

  * Apply limma to identify DEGs (Differentially Expressed Genes).
  
  * Generate volcano plots.
  
  * Filter significant DEGs for downstream ML.

4. Machine Learning (Using Scikit-learn library)

  * Normalize and preprocess data (quantile normalization + z-score).
  
  * Apply variance thresholding and mutual information feature selection.
  
  * Handle class imbalance using SMOTE.
  
  * Train XGBoost classifier with Hyperopt for hyperparameter tuning.
  
  * Evaluate with accuracy, F1-score, ROC-AUC, confusion matrix.

5. Model Interpretation

  * Use SHAP (SHapley Additive exPlanations) for model interpretation:
  
  * Feature importance
  
  * SHAP summary plots
  
  * Waterfall plots

  * Violin plots


🔧 Requirements
R Dependencies

* GEOquery
* data.table
* xml2
* dplyr
* readr
* tidyverse
* limma

Install missing packages in R:
install.packages(c("GEOquery", "data.table", "xml2", "dplyr", "readr", "tidyverse", "limma"))


Python Dependencies:

  * pandas
  * numpy
  * matplotlib
  * shap
  * scikit-learn
  * imbalanced-learn
  * hyperopt
  * xgboost

Install via pip:
pip install pandas numpy matplotlib shap scikit-learn imbalanced-learn hyperopt xgboost
