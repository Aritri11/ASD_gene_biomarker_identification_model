# ============================================================
# STEP 6 — DEG Analysis, Visualization & ML Dataset Preparation
# Key Features:
#    DEG thresholds restored to |logFC|>0.2, adj_pval<0.10
#     (appropriate for blood microarray + limma without SVA)
#   - Design matrix: Condition + Age + Sex (confounders kept)
#   - GSEA used as primary method
#   - GSEA on GO BP and KEGG is used as primary pathway analysis
#     (uses all 16k+ ranked genes, much more powerful for this data)
#   - Full ML-ready dataset with proper labels
# ============================================================

library(tidyverse)
library(limma)
library(pheatmap)
library(ggrepel)
library(GEOquery)
library(clusterProfiler)
library(org.Hs.eg.db)
library(DOSE)
library(dplyr)

# Explicitly bind dplyr functions to avoid namespace conflicts
select    <- dplyr::select
filter    <- dplyr::filter
rename    <- dplyr::rename
mutate    <- dplyr::mutate
arrange   <- dplyr::arrange
summarise <- dplyr::summarise

# ----------------------------
# 6.1 Paths
# ----------------------------
long_format_file <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/merged_data_long_format.csv"
diagnosis_file   <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/GSM_diagnosis.csv"
output_dir       <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ----------------------------
# 6.2 Load Expression Data
# ----------------------------
cat("Loading expression data...\n")
df <- read_csv(long_format_file, show_col_types = FALSE)
cat("✅ Loaded:", nrow(df), "rows.\n\n")

# ----------------------------
# 6.3 Probe Deduplication (highest-variance probe per gene)
#     then Pivot to Wide Format (genes x samples)
# ----------------------------
# Strategy: for genes measured by multiple probes on the Illumina
# HT-12 chip, keep only the probe with the highest variance across
# all samples. This probe captures the most biological variability
# and is most informative for differential expression and ML.
# Averaging (the previous strategy) dilutes signal by mixing
# informative probes with poorly hybridizing ones.
# This is consistent with the deduplication strategy in Step2_improved.R.
# ----------------------------
cat("Selecting highest-variance probe per gene...\n")

library(data.table)

# Work with data.table for efficient per-probe variance calculation
df_clean <- df %>%
  dplyr::filter(!is.na(Symbol), Symbol != "", !is.na(Probe_ID))

# Calculate variance of each probe across all samples
probe_var <- df_clean %>%
  dplyr::group_by(Probe_ID) %>%
  dplyr::summarise(Probe_Variance = var(Expression, na.rm = TRUE),
                   .groups = "drop")

# Attach variance to main data
df_clean <- df_clean %>%
  dplyr::left_join(probe_var, by = "Probe_ID")

# For each gene, keep only the probe with the highest variance
# (one best probe selected globally, then used for all samples)
best_probe_per_gene <- df_clean %>%
  dplyr::group_by(Symbol) %>%
  dplyr::slice_max(order_by = Probe_Variance, n = 1, with_ties = FALSE) %>%
  dplyr::ungroup() %>%
  dplyr::select(Symbol, Probe_ID) %>%
  dplyr::distinct()

cat("✅ Best probe selected for", nrow(best_probe_per_gene), "genes.\n")

# Filter long-format data to keep only the best probe per gene
df_dedup <- df_clean %>%
  dplyr::inner_join(best_probe_per_gene, by = c("Symbol", "Probe_ID")) %>%
  dplyr::select(Symbol, Sample, Expression)

cat("✅ Long-format rows after deduplication:", nrow(df_dedup), "\n\n")

# Pivot to wide format (genes x samples)
cat("Pivoting to wide format...\n")
df_expr <- df_dedup %>%
  pivot_wider(names_from  = Sample,
              values_from = Expression) %>%
  column_to_rownames("Symbol")

cat("✅ Expression matrix:", nrow(df_expr), "genes x",
    ncol(df_expr), "samples.\n\n")

# ----------------------------
# 6.4 Build Sample Metadata (colData)
# ----------------------------
cat("Building sample metadata...\n")

# Get diagnosis labels
conditions <- df %>%
  distinct(Sample, Condition) %>%
  deframe()

# Align columns to condition vector order
df_expr <- df_expr[, names(conditions)]

# Load extended metadata from GEO (for Age and Sex confounders)
cat("Fetching extended metadata from GEO for confounders...\n")
gse <- tryCatch({
  getGEO("GSE42133", GSEMatrix = TRUE)[[1]]
}, error = function(e) {
  cat("⚠️  Could not fetch GEO metadata. Age/Sex will not be used.\n")
  return(NULL)
})

colData <- data.frame(
  Sample    = names(conditions),
  Condition = factor(conditions, levels = c("Control", "ASD")),
  row.names = names(conditions),
  stringsAsFactors = FALSE
)

# Add Age and Sex if GEO metadata available
if (!is.null(gse)) {
  pheno <- pData(gse)
  
  # Align to our sample order
  pheno <- pheno[match(colData$Sample, rownames(pheno)), ]
  
  # Try to extract Age
  age_col <- grep("age", colnames(pheno), ignore.case = TRUE, value = TRUE)[1]
  if (!is.na(age_col)) {
    colData$Age <- suppressWarnings(as.numeric(pheno[[age_col]]))
    cat("✅ Age extracted from column:", age_col, "\n")
  } else {
    colData$Age <- NA
    cat("⚠️  Age column not found in metadata.\n")
  }
  
  # Try to extract Sex
  sex_col <- grep("sex|gender", colnames(pheno), ignore.case = TRUE, value = TRUE)[1]
  if (!is.na(sex_col)) {
    colData$Sex <- factor(pheno[[sex_col]])
    cat("✅ Sex extracted from column:", sex_col, "\n")
  } else {
    colData$Sex <- NA
    cat("⚠️  Sex column not found in metadata.\n")
  }
}

cat("\n=== SAMPLE METADATA SUMMARY ===\n")
cat("Condition distribution:\n")
print(table(colData$Condition))
if (!is.null(gse) && !all(is.na(colData$Age))) {
  cat("Age range:", range(colData$Age, na.rm = TRUE), "\n")
}
cat("\n")

# ----------------------------
# 6.5 Build Design Matrix
# ----------------------------
# NOTE:
# The design matrix retains biological confounders (Age, Sex) which is
# the correct way to handle known covariates without over-correction.
# ----------------------------
cat("Building design matrix (no SVA)...\n")

use_confounders <- !is.null(gse) &&
  !all(is.na(colData$Age)) &&
  !all(is.na(colData$Sex))

if (use_confounders) {
  design_final <- model.matrix(~ Condition + Age + Sex, data = colData)
  cat("✅ Design matrix: Condition + Age + Sex\n")
} else {
  design_final <- model.matrix(~ Condition, data = colData)
  cat("⚠️  Design matrix: Condition only (Age/Sex unavailable)\n")
}

cat("Design matrix columns:", colnames(design_final), "\n")
cat("Residual df:", nrow(colData) - ncol(design_final), "\n\n")

# ----------------------------
# 6.6 Fit limma Model
# ----------------------------
cat("Fitting limma model...\n")

expr_matrix <- as.matrix(df_expr)

fit <- lmFit(expr_matrix, design_final)
fit <- eBayes(fit)

# Extract results for ASD vs Control
limma_results <- topTable(fit,
                          coef    = "ConditionASD",
                          number  = Inf,
                          sort.by = "none") %>%
  tibble::rownames_to_column("Gene") %>%
  dplyr::select(Gene, logFC, AveExpr, t, P.Value, adj.P.Val, B) %>%
  dplyr::rename(Log2FC   = logFC,
                MeanExpr = AveExpr,
                t_stat   = t,
                p_value  = P.Value,
                adj_pval = adj.P.Val,
                B_stat   = B)

cat("✅ limma results computed for", nrow(limma_results), "genes.\n\n")

# Save full DEG results
write_csv(limma_results,
          file.path(output_dir, "DEG_results_full.csv"))
cat("✅ Full DEG results saved: DEG_results_full.csv\n\n")

# ----------------------------
# 6.7 Diagnostics on limma Output
# ----------------------------
cat("=== LIMMA RESULT DIAGNOSTICS ===\n")
cat("Log2FC range:", round(range(limma_results$Log2FC), 4), "\n")
cat("Max absolute Log2FC:", round(max(abs(limma_results$Log2FC)), 4), "\n\n")

cat("Raw p-value summary:\n")
print(summary(limma_results$p_value))
cat("\nAdj p-value summary:\n")
print(summary(limma_results$adj_pval))

cat("\nGene counts at various thresholds:\n")
cat("  |logFC|>0.5, adj_p<0.05:", sum(abs(limma_results$Log2FC)>0.5 & limma_results$adj_pval<0.05), "\n")
cat("  |logFC|>0.3, adj_p<0.05:", sum(abs(limma_results$Log2FC)>0.3 & limma_results$adj_pval<0.05), "\n")
cat("  |logFC|>0.2, adj_p<0.05:", sum(abs(limma_results$Log2FC)>0.2 & limma_results$adj_pval<0.05), "\n")
cat("  |logFC|>0.2, adj_p<0.10:", sum(abs(limma_results$Log2FC)>0.2 & limma_results$adj_pval<0.10), "\n")
cat("  adj_p<0.05 (any FC):    ", sum(limma_results$adj_pval<0.05), "\n")
cat("  adj_p<0.10 (any FC):    ", sum(limma_results$adj_pval<0.10), "\n\n")

cat("Top 20 genes by adjusted p-value:\n")
print(limma_results %>%
        dplyr::arrange(adj_pval) %>%
        head(20) %>%
        dplyr::select(Gene, Log2FC, p_value, adj_pval))

# Log2FC distribution plot
png(file.path(output_dir, "Step6_logFC_distribution.png"), width = 800, height = 500)
hist(limma_results$Log2FC,
     breaks = 100,
     main   = "Distribution of Log2FC values (limma, no SVA)",
     xlab   = "Log2FC",
     col    = "steelblue",
     border = "white")
abline(v = c(-0.2, 0.2), col = "red",  lty = 2, lwd = 2)
abline(v = c(-0.5, 0.5), col = "orange", lty = 3, lwd = 1.5)
legend("topright",
       legend = c("|logFC|=0.2 threshold", "|logFC|=0.5 reference"),
       col    = c("red", "orange"),
       lty    = c(2, 3), lwd = 2)
dev.off()
cat("✅ Log2FC distribution saved.\n\n")

# ----------------------------
# 6.8 Filter Significant DEGs
# ----------------------------
# Thresholds chosen for blood microarray data:
#   |Log2FC| > 0.2 — meaningful effect size for peripheral blood
#   adj_pval < 0.10 — FDR 10%, standard for exploratory microarray studies
# ----------------------------
FC_THRESHOLD   <- 0.2
PVAL_THRESHOLD <- 0.10

sig_degs <- limma_results %>%
  dplyr::filter(abs(Log2FC) > FC_THRESHOLD,
                adj_pval   < PVAL_THRESHOLD) %>%
  dplyr::arrange(adj_pval)

cat("=== SIGNIFICANT DEGs (|logFC|>", FC_THRESHOLD,
    ", adj_pval<", PVAL_THRESHOLD, ") ===\n")
cat("Total:", nrow(sig_degs), "\n")
cat("Upregulated in ASD:", sum(sig_degs$Log2FC > 0), "\n")
cat("Downregulated in ASD:", sum(sig_degs$Log2FC < 0), "\n\n")

# Safety check — warn if too few DEGs for ML
if (nrow(sig_degs) < 80) {
  cat("⚠️  WARNING: Only", nrow(sig_degs),
      "DEGs — consider relaxing thresholds further.\n")
  cat("   SelectKBest(k=80) in Python will fail if fewer than 80 features.\n\n")
} else {
  cat("✅ DEG count is sufficient for ML pipeline (need >80).\n\n")
}

# Top 10 upregulated
cat("Top 10 Upregulated in ASD:\n")
print(sig_degs %>%
        dplyr::filter(Log2FC > 0) %>%
        head(10) %>%
        dplyr::select(Gene, Log2FC, p_value, adj_pval))

# Top 10 downregulated
cat("\nTop 10 Downregulated in ASD:\n")
print(sig_degs %>%
        dplyr::filter(Log2FC < 0) %>%
        head(10) %>%
        dplyr::select(Gene, Log2FC, p_value, adj_pval))

# Save significant DEGs
write_csv(sig_degs,
          file.path(output_dir, "DEG_results_significant.csv"))
cat("\n✅ Significant DEGs saved: DEG_results_significant.csv\n\n")

# ----------------------------
# 6.9 Volcano Plot (with Gene Labels)
# ----------------------------
cat("Generating volcano plot...\n")

volcano_data <- limma_results %>%
  dplyr::mutate(significance = dplyr::case_when(
    Log2FC >  FC_THRESHOLD & adj_pval < PVAL_THRESHOLD ~ "Upregulated in ASD",
    Log2FC < -FC_THRESHOLD & adj_pval < PVAL_THRESHOLD ~ "Downregulated in ASD",
    TRUE                                                ~ "Not Significant"
  ))

# Label top 20 most significant genes
top_label <- limma_results %>%
  dplyr::filter(adj_pval < PVAL_THRESHOLD) %>%
  dplyr::arrange(adj_pval) %>%
  head(20)

volcano_plot <- ggplot(volcano_data,
                       aes(x = Log2FC,
                           y = -log10(adj_pval),
                           color = significance)) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_color_manual(values = c(
    "Upregulated in ASD"   = "red",
    "Downregulated in ASD" = "blue",
    "Not Significant"      = "grey70"
  )) +
  ggrepel::geom_text_repel(
    data         = top_label,
    aes(label    = Gene),
    size         = 3,
    color        = "black",
    max.overlaps = 20,
    segment.color = "grey50"
  ) +
  geom_hline(yintercept = -log10(PVAL_THRESHOLD),
             color = "black", linetype = "dashed") +
  geom_vline(xintercept = c(-FC_THRESHOLD, FC_THRESHOLD),
             color = "black", linetype = "dashed") +
  annotate("text",
           x     = max(abs(limma_results$Log2FC)) * 0.6,
           y     = -log10(PVAL_THRESHOLD) + 0.3,
           label = paste0("FDR = ", PVAL_THRESHOLD * 100, "%"),
           size  = 3) +
  labs(title    = "Volcano Plot — ASD vs Control (limma)",
       subtitle = paste0("Thresholds: |Log2FC| > ", FC_THRESHOLD,
                         ", adj_pval < ", PVAL_THRESHOLD,
                         " | DEGs: ", nrow(sig_degs)),
       x        = "Log2 Fold Change (ASD / Control)",
       y        = "-log10(Adjusted P-value)",
       color    = "Direction") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom",
        plot.title      = element_text(face = "bold"))

ggsave(file.path(output_dir, "volcano_plot_limma.png"),
       plot   = volcano_plot,
       width  = 10,
       height = 7,
       dpi    = 300)
cat("✅ Volcano plot saved.\n\n")

# ----------------------------
# 6.10 Heatmap of Top 50 DEGs
# ----------------------------
cat("Generating heatmap of top DEGs...\n")

n_heatmap <- min(50, nrow(sig_degs))
top_heatmap_genes <- sig_degs %>%
  dplyr::arrange(adj_pval) %>%
  head(n_heatmap) %>%
  dplyr::pull(Gene)

cat("Genes for heatmap:", length(top_heatmap_genes), "\n")

# Extract expression for these genes
expr_heatmap <- expr_matrix[rownames(expr_matrix) %in% top_heatmap_genes, ]
cat("Heatmap matrix dimensions:", dim(expr_heatmap), "\n")

# Align sample order between matrix and annotation
common_samples <- intersect(colnames(expr_heatmap), colData$Sample)
cat("Common samples:", length(common_samples), "\n\n")

expr_heatmap_ordered <- expr_heatmap[, common_samples]

col_annotation <- data.frame(
  Condition = colData$Condition[match(common_samples, colData$Sample)],
  row.names = common_samples
)

ann_colors <- list(
  Condition = c(ASD = "#E41A1C", Control = "#377EB8")
)

# Verify alignment
cat("Alignment check — cols match annotation rows:",
    all(colnames(expr_heatmap_ordered) == rownames(col_annotation)), "\n")

# Remove zero-variance genes
row_vars <- apply(expr_heatmap_ordered, 1, var, na.rm = TRUE)
cat("Genes with zero variance:", sum(row_vars == 0, na.rm = TRUE), "\n")
expr_heatmap_ordered <- expr_heatmap_ordered[!is.na(row_vars) & row_vars > 0, ]
cat("Genes retained for heatmap:", nrow(expr_heatmap_ordered), "\n\n")

heatmap_path <- file.path(output_dir, "heatmap_top_DEGs.png")

pheatmap::pheatmap(
  mat               = expr_heatmap_ordered,
  annotation_col    = col_annotation,
  annotation_colors = ann_colors,
  scale             = "row",
  show_rownames     = TRUE,
  show_colnames     = FALSE,
  color             = colorRampPalette(c("navy", "white", "firebrick3"))(100),
  clustering_method          = "ward.D2",
  clustering_distance_rows   = "euclidean",
  clustering_distance_cols   = "euclidean",
  main              = paste0("Top ", nrow(expr_heatmap_ordered),
                             " DEGs — ASD vs Control (Z-scored)"),
  fontsize          = 10,
  fontsize_row      = 7,
  border_color      = NA,
  filename          = heatmap_path,
  width             = 12,
  height            = 10
)

size_kb <- file.size(heatmap_path) / 1024
cat("Heatmap file size:", round(size_kb, 1), "KB\n")
if (size_kb > 50) {
  cat("✅ Heatmap looks healthy (>50KB).\n\n")
} else if (size_kb > 5) {
  cat("⚠️  File saved but small — check the image.\n\n")
} else {
  cat("❌ File too small — displaying in RStudio plot panel instead...\n")
  pheatmap::pheatmap(
    mat               = expr_heatmap_ordered,
    annotation_col    = col_annotation,
    annotation_colors = ann_colors,
    scale             = "row",
    show_rownames     = TRUE,
    show_colnames     = FALSE,
    color             = colorRampPalette(c("navy", "white", "firebrick3"))(100),
    clustering_method = "ward.D2",
    main              = "Top DEGs — ASD vs Control"
  )
  cat("Use RStudio Plots panel → Export → Save as Image → PNG\n\n")
}

# ----------------------------
# 6.11 Pathway Enrichment Analysis — GSEA (primary method)
# ----------------------------
# GSEA uses ALL 16k+ genes ranked by t-statistic and is far more
# powerful for subtle blood microarray effects, making it the clear primary method.
# ----------------------------
cat("Running GSEA pathway enrichment analysis...\n\n")

# Build ranked gene list from all characterised genes
# Use t-statistic (captures both effect size and significance)
ranked_genes <- limma_results %>%
  dplyr::filter(!grepl("^LOC|^LINC|^MIR|^SNOR", Gene)) %>%
  dplyr::arrange(dplyr::desc(t_stat))

# Map gene symbols to Entrez IDs
ranked_entrez <- bitr(ranked_genes$Gene,
                      fromType = "SYMBOL",
                      toType   = "ENTREZID",
                      OrgDb    = org.Hs.eg.db)

# Merge t-statistics with Entrez IDs
ranked_merged <- merge(ranked_genes, ranked_entrez,
                       by.x = "Gene", by.y = "SYMBOL")

# Create named numeric vector: Entrez ID -> t-statistic
gene_list <- ranked_merged$t_stat
names(gene_list) <- ranked_merged$ENTREZID
gene_list <- sort(gene_list, decreasing = TRUE)
gene_list <- gene_list[!duplicated(names(gene_list))]

cat("Ranked gene list size:", length(gene_list), "\n\n")

# --- GSEA on GO Biological Process ---
cat("Running GSEA on GO Biological Process...\n")

gsea_go <- tryCatch({
  gseGO(geneList     = gene_list,
        OrgDb        = org.Hs.eg.db,
        ont          = "BP",
        minGSSize    = 10,
        maxGSSize    = 500,
        pvalueCutoff = 0.05,
        verbose      = FALSE)
}, error = function(e) {
  cat("⚠️  GSEA GO BP failed:", e$message, "\n")
  return(NULL)
})

if (!is.null(gsea_go) && nrow(gsea_go) > 0) {
  cat("✅ GSEA GO BP:", nrow(gsea_go), "significant gene sets.\n")
  
  write_csv(as.data.frame(gsea_go),
            file.path(output_dir, "GSEA_GO_BP_results.csv"))
  cat("✅ GSEA GO BP results saved: GSEA_GO_BP_results.csv\n")
  
  gsea_go_plot <- dotplot(gsea_go,
                          showCategory = min(20, nrow(gsea_go)),
                          split        = ".sign",
                          title        = "GSEA GO Biological Process — ASD vs Control") +
    facet_grid(. ~ .sign) +
    theme(axis.text.y = element_text(size = 7))
  
  ggsave(file.path(output_dir, "GSEA_GO_BP_dotplot.png"),
         plot   = gsea_go_plot,
         width  = 12,
         height = 8,
         dpi    = 300)
  cat("✅ GSEA GO BP dotplot saved.\n\n")
  
} else {
  cat("⚠️  No significant GSEA GO BP gene sets.\n\n")
}

# --- GSEA on KEGG ---
cat("Running GSEA on KEGG pathways...\n")

gsea_kegg <- tryCatch({
  gseKEGG(geneList     = gene_list,
          organism     = "hsa",
          minGSSize    = 10,
          pvalueCutoff = 0.05,
          verbose      = FALSE)
}, error = function(e) {
  cat("⚠️  GSEA KEGG failed:", e$message, "\n")
  return(NULL)
})

if (!is.null(gsea_kegg) && nrow(gsea_kegg) > 0) {
  cat("✅ GSEA KEGG:", nrow(gsea_kegg), "significant pathways.\n")
  
  write_csv(as.data.frame(gsea_kegg),
            file.path(output_dir, "GSEA_KEGG_results.csv"))
  cat("✅ GSEA KEGG results saved: GSEA_KEGG_results.csv\n")
  
  gsea_kegg_plot <- dotplot(gsea_kegg,
                            showCategory = min(20, nrow(gsea_kegg)),
                            split        = ".sign",
                            title        = "GSEA KEGG Pathways — ASD vs Control") +
    facet_grid(. ~ .sign) +
    theme(axis.text.y = element_text(size = 7))
  
  ggsave(file.path(output_dir, "GSEA_KEGG_dotplot.png"),
         plot   = gsea_kegg_plot,
         width  = 12,
         height = 8,
         dpi    = 300)
  cat("✅ GSEA KEGG dotplot saved.\n\n")
  
} else {
  cat("⚠️  No significant GSEA KEGG pathways.\n\n")
}

# ----------------------------
# 6.12 Build ML-Ready Dataset
# ----------------------------
cat("Building ML-ready dataset...\n")

ml_genes_final <- sig_degs$Gene

expr_ml   <- expr_matrix[rownames(expr_matrix) %in% ml_genes_final, , drop = FALSE]
expr_ml_t <- t(expr_ml)

sample_order  <- rownames(expr_ml_t)
condition_vec <- colData$Condition[match(sample_order, colData$Sample)]

ml_data <- data.frame(
  Sample    = sample_order,
  Condition = as.character(condition_vec),
  Label     = ifelse(as.character(condition_vec) == "ASD", 1, 0),
  expr_ml_t,
  check.names = TRUE
)

cat("ML dataset dimensions:", nrow(ml_data), "samples x",
    ncol(ml_data) - 3, "gene features.\n")
cat("Class distribution:\n")
print(table(ml_data$Condition))

# Final safety check for Python pipeline compatibility
n_features <- ncol(ml_data) - 3
if (n_features < 80) {
  cat("\n⚠️  WARNING:", n_features, "features < 80.\n")
  cat("   Update SelectKBest k in Python to:", floor(n_features * 0.6), "\n")
} else {
  cat("\n✅ Feature count", n_features, ">= 80. SelectKBest(k=80) will work.\n")
  cat("   Recommended k = min(80, floor(", n_features, "* 0.5)) =",
      min(80, floor(n_features * 0.5)), "\n")
}

write_csv(ml_data, file.path(output_dir, "ML_ready_dataset.csv"))
cat("✅ ML dataset saved: ML_ready_dataset.csv\n\n")

# ----------------------------
# Final Summary
# ----------------------------
cat("\n========================================\n")
cat("STEP 6 COMPLETE — ANALYSIS SUMMARY\n")
cat("========================================\n")
cat("Total genes tested:         ", nrow(limma_results), "\n")
cat("Design matrix:              ", paste(colnames(design_final), collapse=" + "), "\n")
cat("DEG thresholds:              |logFC|>", FC_THRESHOLD,
    ", adj_pval<", PVAL_THRESHOLD, "\n")
cat("Significant DEGs:           ", nrow(sig_degs), "\n")
cat("  Upregulated in ASD:       ", sum(sig_degs$Log2FC > 0), "\n")
cat("  Downregulated in ASD:     ", sum(sig_degs$Log2FC < 0), "\n")
cat("ML features (gene count):   ", n_features, "\n")
cat("ML samples:                 ", nrow(ml_data), "\n")
cat("  ASD:                      ", sum(ml_data$Condition == "ASD"), "\n")
cat("  Control:                  ", sum(ml_data$Condition == "Control"), "\n")
cat("Pathway method:              GSEA \n")
cat("Outputs saved to:           ", output_dir, "\n")
cat("========================================\n")
