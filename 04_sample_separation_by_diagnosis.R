# ============================================================
# STEP 4 — Separate Samples into ASD and Control Folders
# Key Features: Error handling, progress tracking, summary report
# ============================================================

library(data.table)

# ----------------------------
# 4.1 Paths — UPDATE THESE
# ----------------------------
annot_path         <- "C:/Users/Aritri Baidya/Downloads/GPL10558-50081.txt"
input_folder       <- "C:/Users/Aritri Baidya/Desktop/ML Project/GSE42133_family"
diagnosis_csv      <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/GSM_diagnosis.csv"
output_folder_asd  <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/ASD_2"
output_folder_ctrl <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/Controls_2"

# ----------------------------
# 4.2 Create Output Directories
# ----------------------------
dir.create(output_folder_asd,  showWarnings = FALSE, recursive = TRUE)
dir.create(output_folder_ctrl, showWarnings = FALSE, recursive = TRUE)
cat("✅ Output directories created/verified.\n\n")

# ----------------------------
# 4.3 Load Annotation File
# ----------------------------
cat("Loading annotation file...\n")
annot_df <- fread(annot_path,
                  sep = "\t",
                  header = TRUE,
                  skip = "#",
                  data.table = FALSE)

annot_df <- annot_df[, c("ID", "Symbol")]

# Remove probes with missing gene symbols
annot_df <- annot_df[!is.na(annot_df$Symbol) &
                       annot_df$Symbol != "" &
                       annot_df$Symbol != "---", ]

cat("✅ Annotation loaded:", nrow(annot_df), "valid probes.\n\n")

# ----------------------------
# 4.4 Load Diagnosis Mapping
# ----------------------------
cat("Loading diagnosis mapping...\n")
mapping <- fread(diagnosis_csv, data.table = FALSE)

# Validate columns exist
if (!all(c("GSM_ID", "Diagnosis") %in% colnames(mapping))) {
  stop("❌ GSM_diagnosis.csv must have columns: GSM_ID, Diagnosis")
}

cat("✅ Mapping loaded:", nrow(mapping), "samples.\n")
cat("  ASD:", sum(mapping$Diagnosis == "ASD"), "\n")
cat("  Control:", sum(mapping$Diagnosis == "Control"), "\n\n")

# ----------------------------
# 4.5 Process Each GSM File
# ----------------------------
# Tracking counters
success_asd  <- 0
success_ctrl <- 0
missing      <- c()
errors       <- c()
skipped      <- c()

cat("Processing GSM files...\n")
cat("----------------------------------------\n")

for (i in 1:nrow(mapping)) {

  gsm_id    <- mapping$GSM_ID[i]
  diagnosis <- mapping$Diagnosis[i]

  # Skip if diagnosis is not ASD or Control
  if (!diagnosis %in% c("ASD", "Control")) {
    cat("⏭️  Skipping", gsm_id, "— unknown diagnosis:", diagnosis, "\n")
    skipped <- c(skipped, gsm_id)
    next
  }

  # Construct input file path
  filename   <- paste0(gsm_id, "-tbl-1.txt")
  input_path <- file.path(input_folder, filename)

  # Check if file exists
  if (!file.exists(input_path)) {
    cat("❌ Missing file:", filename, "\n")
    missing <- c(missing, gsm_id)
    next
  }

  # Determine output path
  if (diagnosis == "ASD") {
    output_path <- file.path(output_folder_asd,
                              paste0("expression_data_", gsm_id, ".csv"))
  } else {
    output_path <- file.path(output_folder_ctrl,
                              paste0("expression_data_", gsm_id, ".csv"))
  }

  # Load, merge, and save with full error handling
  tryCatch({
    # Load expression data
    expr_df <- fread(input_path,
                     sep = "\t",
                     header = FALSE,
                     data.table = FALSE)
    colnames(expr_df) <- c("Probe_ID", "Expression")

    # Validate columns
    if (ncol(expr_df) < 2) {
      stop("File has fewer than 2 columns")
    }

    # Check for empty data
    if (nrow(expr_df) == 0) {
      stop("File is empty")
    }

    # Merge with annotation
    merged_df <- merge(expr_df, annot_df,
                       by.x = "Probe_ID",
                       by.y = "ID",
                       all.x = TRUE)

    # Select and order columns
    merged_df <- merged_df[, c("Probe_ID", "Symbol", "Expression")]

    # Remove rows with no gene symbol
    merged_df <- merged_df[!is.na(merged_df$Symbol) &
                             merged_df$Symbol != "", ]

    # Save
    fwrite(merged_df, output_path)

    # Update counters
    if (diagnosis == "ASD") {
      success_asd <- success_asd + 1
    } else {
      success_ctrl <- success_ctrl + 1
    }

    # Progress indicator every 10 samples
    if ((success_asd + success_ctrl) %% 10 == 0) {
      cat("  Processed", success_asd + success_ctrl, "samples so far...\n")
    }

  }, error = function(e) {
    cat("❌ Error processing", gsm_id, ":", conditionMessage(e), "\n")
    errors <<- c(errors, gsm_id)
  })
}

# ----------------------------
# 4.6 Summary Report
# ----------------------------
cat("\n========================================\n")
cat("STEP 4 COMPLETE — PROCESSING SUMMARY\n")
cat("========================================\n")
cat("✅ Successfully saved ASD files:     ", success_asd, "\n")
cat("✅ Successfully saved Control files: ", success_ctrl, "\n")

if (length(missing) > 0) {
  cat("❌ Missing input files (", length(missing), "):\n")
  cat(paste(" ", missing, collapse = "\n"), "\n")
}

if (length(errors) > 0) {
  cat("❌ Files with processing errors (", length(errors), "):\n")
  cat(paste(" ", errors, collapse = "\n"), "\n")
}

if (length(skipped) > 0) {
  cat("⏭️  Skipped (unknown diagnosis) (", length(skipped), "):\n")
  cat(paste(" ", skipped, collapse = "\n"), "\n")
}

cat("\nOutput folders:\n")
cat("  ASD:     ", output_folder_asd, "\n")
cat("  Control: ", output_folder_ctrl, "\n")
cat("========================================\n")
