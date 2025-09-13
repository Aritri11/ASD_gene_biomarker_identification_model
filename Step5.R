library(dplyr)
library(readr)

# === Paths ===
metadata_file <- "C:/Users/Aritri Baidya/Desktop/ML Project/GSM_diagnosis.csv"
asd_folder <- "C:/Users/Aritri Baidya/Desktop/ML Project/ASD_2"
control_folder <- "C:/Users/Aritri Baidya/Desktop/ML Project/Controls_2"
output_file <- "C:/Users/Aritri Baidya/Desktop/ML Project/merged_data_long_format.csv"

# === Load sample metadata (GSM_ID, Diagnosis) ===
metadata <- read_csv(metadata_file)

# === Function to load one sample ===
load_sample <- function(gsm_id, diagnosis) {
  # Build file path depending on diagnosis
  folder <- ifelse(diagnosis == "ASD", asd_folder, control_folder)
  file_path <- file.path(folder, paste0("expression_data_", gsm_id, ".csv"))
  
  if (!file.exists(file_path)) {
    warning(paste("File not found:", file_path))
    return(NULL)
  }
  
  # Load CSV
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # Add sample metadata
  df <- df %>%
    mutate(Sample = gsm_id, Condition = diagnosis)
  
  return(df)
}

# === Apply to all samples ===
all_data <- lapply(1:nrow(metadata), function(i) {
  load_sample(metadata$GSM_ID[i], metadata$Diagnosis[i])
})

# === Combine into one big dataframe ===
merged_df <- bind_rows(all_data)

# === Reorder columns ===
merged_df <- merged_df %>%
  select(Probe_ID, Symbol, Expression, Sample, Condition)

# === Save output ===
write_csv(merged_df, output_file)

cat("✅ Merged expression data saved to:", output_file, "\n")
