library(data.table)

# === 1. Load GPL10558 Annotation File ===
annot_df <- fread("C:/Users/Aritri Baidya/Downloads/GPL10558-50081.txt",
                  sep = "\t", header = TRUE, skip = "#", data.table = FALSE)

# Keep only Probe ID and Gene Symbol
annot_df <- annot_df[, c("ID", "Symbol")]

# === 2. List all GSM expression files ===
files <- list.files("C:/Users/Aritri Baidya/Desktop/ML Project/GSE42133_family",
                    pattern = "GSM.*-tbl-1.txt$", full.names = TRUE)

cat("Found", length(files), "files\n")  # should print 147

# === 3. Process each GSM file ===
expr_list <- list()

for (f in files) {
  # Read expression data
  expr_df <- fread(f, sep = "\t", header = FALSE)
  colnames(expr_df) <- c("Probe_ID", "Expression")
  
  # Merge with annotation
  merged_df <- merge(expr_df, annot_df, by.x = "Probe_ID", by.y = "ID", all.x = TRUE)
  
  # Use GSM ID (from filename) as column name
  gsm_id <- sub("-tbl-1.txt", "", basename(f))   # e.g., GSM1033110
  merged_df <- merged_df[, c("Probe_ID", "Symbol", "Expression")]
  colnames(merged_df)[3] <- gsm_id
  
  # Save into list
  expr_list[[gsm_id]] <- merged_df
}

# === 4. Combine all samples into one matrix ===
# Start with first data frame
final_df <- expr_list[[1]]

# Merge iteratively with the rest
for (i in 2:length(expr_list)) {
  final_df <- merge(final_df, expr_list[[i]], by = c("Probe_ID", "Symbol"), all = TRUE)
}

# === 5. Save combined matrix ===
write.csv(final_df,
          "C:/Users/Aritri Baidya/Desktop/ML Project/R_codes/expression_matrix_all_samples.csv",
          row.names = FALSE)

cat("✅ Done! Final matrix saved with", ncol(final_df)-2, "samples\n")
