# ============================================================
# STEP 2 — Build Expression Matrix from Raw GSM Files
# Key Features: Fast merging, probe filtering, duplicate handling
# ============================================================

library(data.table)
library(dplyr)

# ----------------------------
# 2.1 Paths — UPDATE THESE
# ----------------------------
annot_path   <- "C:/Users/Aritri Baidya/Downloads/GPL10558-50081.txt"
gsm_folder   <- "C:/Users/Aritri Baidya/Desktop/ML Project/GSE42133_family"
output_path  <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/expression_matrix_all_samples.csv"

# ----------------------------
# 2.2 Load Annotation File
# ----------------------------
cat("Loading annotation file...\n")
annot_df <- fread(annot_path,
                  sep = "\t",
                  header = TRUE,
                  skip = "#",
                  data.table = FALSE)

# Keep only Probe ID and Gene Symbol
annot_df <- annot_df[, c("ID", "Symbol")]

# Remove probes with missing or empty gene symbols
annot_df <- annot_df %>%
  filter(!is.na(Symbol), Symbol != "", Symbol != "---")

cat("✅ Annotation loaded:", nrow(annot_df), "probes with valid gene symbols.\n\n")

# ----------------------------
# 2.3 List All GSM Files
# ----------------------------
files <- list.files(gsm_folder,
                    pattern = "GSM.*-tbl-1\\.txt$",
                    full.names = TRUE)

cat("Found", length(files), "GSM expression files.\n")
if (length(files) != 147) {
  warning(paste("⚠️  Expected 147 files, found:", length(files)))
}

# ----------------------------
# 2.4 Fast Loading Using rbindlist (replaces slow iterative merge)
# ----------------------------
cat("\nLoading all GSM files — this may take a few minutes...\n")

# Load all files into a single long-format data.table at once
all_expr_list <- lapply(files, function(f) {
  dt <- tryCatch({
    fread(f, sep = "\t", header = FALSE,
          col.names = c("Probe_ID", "Expression"))
  }, error = function(e) {
    cat("❌ Error reading:", basename(f), "-", e$message, "\n")
    return(NULL)
  })

  if (is.null(dt)) return(NULL)

  # Extract GSM ID from filename
  gsm_id <- sub("-tbl-1\\.txt$", "", basename(f))
  dt[, Sample := gsm_id]
  return(dt)
})

# Remove any NULLs (failed files)
failed <- sum(sapply(all_expr_list, is.null))
if (failed > 0) cat("⚠️", failed, "files failed to load.\n")

all_expr_list <- Filter(Negate(is.null), all_expr_list)

# Combine all samples into one long-format table
cat("Combining all samples...\n")
long_dt <- rbindlist(all_expr_list)
cat("✅ Combined long table:", nrow(long_dt), "rows,",
    length(unique(long_dt$Sample)), "samples.\n\n")

# ----------------------------
# 2.5 Merge with Annotation
# ----------------------------
cat("Merging with annotation...\n")
long_dt <- merge(long_dt, annot_df,
                 by.x = "Probe_ID",
                 by.y = "ID",
                 all.x = TRUE)

# Remove probes not in annotation
long_dt <- long_dt[!is.na(Symbol) & Symbol != ""]
cat("✅ After annotation merge:", length(unique(long_dt$Probe_ID)),
    "probes retained.\n\n")

# ----------------------------
# 2.6 Handle Duplicate Probes per Gene
# Strategy: Keep probe with highest variance across samples
#           (captures most biological signal)
# ----------------------------
cat("Handling duplicate probes per gene...\n")

# Calculate variance per probe across all samples
probe_var <- long_dt[, .(Variance = var(Expression, na.rm = TRUE)),
                     by = Probe_ID]
long_dt <- merge(long_dt, probe_var, by = "Probe_ID")

# For each gene symbol, keep only the probe with highest variance
best_probes <- long_dt[, .SD[which.max(Variance)], by = .(Symbol, Sample)]
best_probes[, Variance := NULL]  # remove helper column

cat("✅ After deduplication:", length(unique(best_probes$Symbol)),
    "unique genes retained.\n\n")

# ----------------------------
# 2.7 Pivot to Wide Format (genes × samples)
# ----------------------------
cat("Pivoting to wide format (genes × samples)...\n")

expr_wide <- dcast(best_probes,
                   Symbol ~ Sample,
                   value.var = "Expression")

cat("✅ Wide matrix dimensions:", nrow(expr_wide), "genes ×",
    ncol(expr_wide) - 1, "samples.\n\n")

# ----------------------------
# 2.8 Basic QC on Final Matrix
# ----------------------------
expr_matrix_only <- as.matrix(expr_wide[, -1])
rownames(expr_matrix_only) <- expr_wide$Symbol

cat("=== FINAL MATRIX QC ===\n")
cat("Value range:", round(range(expr_matrix_only, na.rm = TRUE), 3), "\n")
cat("Missing values:", sum(is.na(expr_matrix_only)), "\n")
cat("Genes:", nrow(expr_matrix_only), "\n")
cat("Samples:", ncol(expr_matrix_only), "\n\n")

# ----------------------------
# 2.9 Save Outputs
# ----------------------------
fwrite(expr_wide, output_path)
cat("✅ Wide expression matrix saved:", output_path, "\n")

# Also save as RDS for fast reloading in R
saveRDS(expr_matrix_only, "Step2_expression_matrix_wide.rds")
cat("✅ RDS saved: Step2_expression_matrix_wide.rds\n")

cat("\n========================================\n")
cat("STEP 2 COMPLETE\n")
cat("Outputs:\n")
cat("  - expression_matrix_all_samples.csv (", nrow(expr_wide),
    "genes ×", ncol(expr_wide)-1, "samples )\n")
cat("  - Step2_expression_matrix_wide.rds\n")
cat("========================================\n")
