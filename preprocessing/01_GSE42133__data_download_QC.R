# ============================================================
# STEP 1 — Data Loading, QC & Exploration
# Dataset: GSE42133 (ASD blood microarray — Illumina HT-12 V4)
# ============================================================

library(GEOquery)
library(ggplot2)

# ----------------------------
# 1.1 Download & Load GEO Data
# ----------------------------
cat("Downloading GSE42133 from GEO...\n")
gse <- getGEO("GSE42133", GSEMatrix = TRUE)[[1]]
cat("✅ Data loaded successfully.\n\n")

# ----------------------------
# 1.2 Basic Dataset Information
# ----------------------------
cat("=== DATASET SUMMARY ===\n")
cat("Number of probes (rows):", nrow(exprs(gse)), "\n")
cat("Number of samples (cols):", ncol(exprs(gse)), "\n\n")

# ----------------------------
# 1.3 Extract Expression Matrix
# ----------------------------
expr <- exprs(gse)   # rows = probes, cols = samples

# --- Validate value range ---
val_range <- range(expr, na.rm = TRUE)
cat("=== EXPRESSION VALUE QC ===\n")
cat("Min value:", round(val_range[1], 3), "\n")
cat("Max value:", round(val_range[2], 3), "\n")
cat("Mean value:", round(mean(expr, na.rm = TRUE), 3), "\n")

# Expected: log2 normalized values should be roughly 4–14
if (val_range[1] >= 0 && val_range[2] <= 20) {
  cat("✅ Values are in expected log2 range (0–20).\n")
} else {
  cat("⚠️  WARNING: Values outside expected log2 range — check normalization!\n")
}

# --- Check for missing values ---
na_count <- sum(is.na(expr))
cat("\nMissing values (NA):", na_count, "\n")
if (na_count == 0) {
  cat("✅ No missing values found.\n")
} else {
  cat("⚠️  WARNING:", na_count, "missing values detected.\n")
}

# --- Check for negative values ---
neg_count <- sum(expr < 0, na.rm = TRUE)
cat("Negative values:", neg_count, "\n")
if (neg_count == 0) {
  cat("✅ No negative values found.\n\n")
} else {
  cat("⚠️  WARNING:", neg_count, "negative values — unexpected after log2 normalization.\n\n")
}

# ----------------------------
# 1.4 QC Plots
# ----------------------------

# --- Histogram of all expression values ---
png("Step1_histogram_expression.png", width = 800, height = 600)
hist(as.vector(expr),
     breaks = 100,
     main = "Distribution of Expression Values (All Probes, All Samples)",
     xlab = "Log2 Expression",
     col = "steelblue",
     border = "white")
abline(v = mean(expr, na.rm = TRUE), col = "red", lwd = 2, lty = 2)
legend("topright", legend = paste("Mean =", round(mean(expr, na.rm=TRUE), 2)),
       col = "red", lty = 2, lwd = 2)
dev.off()
cat("✅ Histogram saved: Step1_histogram_expression.png\n")

# --- Boxplot of first 30 samples (check uniform distributions = quantile normalized) ---
png("Step1_boxplot_samples.png", width = 1200, height = 600)
boxplot(expr[, 1:min(30, ncol(expr))],
        las = 2,
        col = "lightblue",
        main = "Sample Distributions (First 30 Samples) — Should Be Uniform After Quantile Normalization",
        ylab = "Log2 Expression",
        cex.axis = 0.6)
dev.off()
cat("✅ Boxplot saved: Step1_boxplot_samples.png\n\n")

# ----------------------------
# 1.5 Extract & Explore Metadata
# ----------------------------
meta <- pData(gse)

cat("=== METADATA COLUMNS AVAILABLE ===\n")
print(colnames(meta))
cat("\n")

# --- Show key metadata columns ---
cat("=== SAMPLE METADATA PREVIEW ===\n")
key_cols <- colnames(meta)[grepl("title|source|characteristics|geo_accession",
                                  colnames(meta), ignore.case = TRUE)]
print(head(meta[, key_cols], 10))

# --- Check for diagnosis column ---
cat("\n=== DIAGNOSIS DISTRIBUTION ===\n")
# Try common column names for diagnosis
diag_col <- grep("characteristics_ch1", colnames(meta), value = TRUE)
if (length(diag_col) > 0) {
  for (col in diag_col) {
    cat("\nColumn:", col, "\n")
    print(table(meta[[col]]))
  }
}

# ----------------------------
# 1.6 Save Metadata
# ----------------------------
write.csv(meta,
          "Step1_sample_metadata.csv",
          row.names = TRUE)
cat("\n✅ Full metadata saved: Step1_sample_metadata.csv\n")

# ----------------------------
# 1.7 Save Expression Matrix
# ----------------------------
saveRDS(expr, "Step1_expression_matrix.rds")
cat("✅ Expression matrix saved as RDS: Step1_expression_matrix.rds\n")

cat("\n========================================\n")
cat("STEP 1 COMPLETE\n")
cat("Outputs:\n")
cat("  - Step1_histogram_expression.png\n")
cat("  - Step1_boxplot_samples.png\n")
cat("  - Step1_sample_metadata.csv\n")
cat("  - Step1_expression_matrix.rds\n")
cat("========================================\n")
