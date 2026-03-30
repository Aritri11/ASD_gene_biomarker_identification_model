# ============================================================
# STEP 5 — Merge All Samples into Wide Format Matrix
# Key Features: Wide format (memory efficient), validation,
#               duplicate probe handling, missing file tracking
# ============================================================

library(data.table)
library(dplyr)

# ----------------------------
# 5.1 Paths — UPDATE THESE
# ----------------------------
metadata_file   <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/GSM_diagnosis.csv"
asd_folder      <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/ASD_2"
control_folder  <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/Controls_2"
output_wide     <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/merged_expression_wide.csv"
output_long     <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/merged_data_long_format.csv"

# ----------------------------
# 5.2 Load Metadata
# ----------------------------
cat("Loading metadata...\n")
metadata <- fread(metadata_file, data.table = FALSE)

cat("✅ Metadata loaded:", nrow(metadata), "samples.\n")
cat("  ASD:", sum(metadata$Diagnosis == "ASD"), "\n")
cat("  Control:", sum(metadata$Diagnosis == "Control"), "\n\n")

# ----------------------------
# 5.3 Function to Load One Sample
# ----------------------------
load_sample <- function(gsm_id, diagnosis) {

  # Build file path
  folder    <- ifelse(diagnosis == "ASD", asd_folder, control_folder)
  file_path <- file.path(folder, paste0("expression_data_", gsm_id, ".csv"))

  # Check existence
  if (!file.exists(file_path)) {
    warning(paste("⚠️  File not found:", file_path))
    return(NULL)
  }

  # Load with error handling
  df <- tryCatch({
    fread(file_path, data.table = FALSE)
  }, error = function(e) {
    warning(paste("❌ Error loading", gsm_id, ":", e$message))
    return(NULL)
  })

  if (is.null(df)) return(NULL)

  # Validate expected columns
  if (!all(c("Probe_ID", "Symbol", "Expression") %in% colnames(df))) {
    warning(paste("⚠️  Unexpected columns in", gsm_id))
    return(NULL)
  }

  # Add sample metadata
  df$Sample    <- gsm_id
  df$Condition <- diagnosis

  return(df)
}

# ----------------------------
# 5.4 Load All Samples (Long Format First)
# ----------------------------
cat("Loading all sample files...\n")

all_data <- lapply(1:nrow(metadata), function(i) {
  load_sample(metadata$GSM_ID[i], metadata$Diagnosis[i])
})

# Track missing files
missing_idx <- which(sapply(all_data, is.null))
if (length(missing_idx) > 0) {
  cat("⚠️  Could not load", length(missing_idx), "samples:\n")
  cat(paste(" ", metadata$GSM_ID[missing_idx], collapse = "\n"), "\n\n")
}

# Remove NULLs
all_data <- Filter(Negate(is.null), all_data)
cat("✅ Successfully loaded", length(all_data), "samples.\n\n")

# ----------------------------
# 5.5 Combine into Long Format
# ----------------------------
cat("Combining into long format...\n")
merged_long <- rbindlist(all_data, fill = TRUE)
merged_long <- merged_long %>%
  select(Probe_ID, Symbol, Expression, Sample, Condition)

cat("✅ Long format dimensions:", nrow(merged_long), "rows ×",
    ncol(merged_long), "cols.\n")
cat("  Unique probes:", length(unique(merged_long$Probe_ID)), "\n")
cat("  Unique genes:", length(unique(merged_long$Symbol[!is.na(merged_long$Symbol)])), "\n")
cat("  Unique samples:", length(unique(merged_long$Sample)), "\n\n")

# ----------------------------
# 5.6 Save Long Format (for compatibility with Step 6)
# ----------------------------
fwrite(merged_long, output_long)
cat("✅ Long format saved:", output_long, "\n\n")

# ----------------------------
# 5.7 Create Wide Format (genes × samples) — more memory efficient
# ----------------------------
cat("Creating wide format matrix (genes × samples)...\n")

# Remove probes with no gene symbol
merged_long_clean <- merged_long %>%
  filter(!is.na(Symbol), Symbol != "")

# Average multiple probes per gene per sample
gene_avg <- merged_long_clean %>%
  group_by(Symbol, Sample) %>%
  summarise(Expression = mean(Expression, na.rm = TRUE),
            .groups = "drop")

# Pivot to wide format
wide_dt <- dcast(setDT(gene_avg),
                 Symbol ~ Sample,
                 value.var = "Expression")

cat("✅ Wide matrix dimensions:", nrow(wide_dt), "genes ×",
    ncol(wide_dt) - 1, "samples.\n\n")

# ----------------------------
# 5.8 QC on Wide Matrix
# ----------------------------
expr_mat <- as.matrix(wide_dt[, -1])
rownames(expr_mat) <- wide_dt$Symbol

cat("=== WIDE MATRIX QC ===\n")
cat("Value range:", round(range(expr_mat, na.rm = TRUE), 3), "\n")
cat("Missing values:", sum(is.na(expr_mat)), "\n")

# Check that samples match metadata
missing_from_wide <- setdiff(metadata$GSM_ID, colnames(expr_mat))
extra_in_wide     <- setdiff(colnames(expr_mat), metadata$GSM_ID)

if (length(missing_from_wide) == 0) {
  cat("✅ All metadata samples present in wide matrix.\n")
} else {
  cat("⚠️  Samples in metadata but not in matrix:",
      paste(missing_from_wide, collapse = ", "), "\n")
}

# ----------------------------
# 5.9 Save Wide Format
# ----------------------------
fwrite(wide_dt, output_wide)
cat("\n✅ Wide format saved:", output_wide, "\n")

# Save condition vector aligned to wide matrix columns
condition_vec <- metadata$Diagnosis[match(colnames(expr_mat), metadata$GSM_ID)]
names(condition_vec) <- colnames(expr_mat)
saveRDS(condition_vec, "Step5_condition_vector.rds")
cat("✅ Condition vector saved: Step5_condition_vector.rds\n")

# Save wide matrix as RDS for fast loading
saveRDS(expr_mat, "Step5_expression_wide_matrix.rds")
cat("✅ Wide matrix RDS saved: Step5_expression_wide_matrix.rds\n")

cat("\n========================================\n")
cat("STEP 5 COMPLETE\n")
cat("Outputs:\n")
cat("  - merged_data_long_format.csv\n")
cat("  - merged_expression_wide.csv\n")
cat("  - Step5_expression_wide_matrix.rds\n")
cat("  - Step5_condition_vector.rds\n")
cat("========================================\n")
