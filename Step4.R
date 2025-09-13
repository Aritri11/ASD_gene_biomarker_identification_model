library(data.table)

# === Path to Annotation File ===
annot_path <- "C:/Users/Aritri Baidya/Downloads/GPL10558-50081.txt"

# === Folder with Expression Files ===
input_folder <- "C:/Users/Aritri Baidya/Desktop/ML Project/GSE42133_family"

# === Output Folders for ASD and Control ===
output_folder_asd <- "C:/Users/Aritri Baidya/Desktop/ML Project/ASD_2"
output_folder_ctrl <- "C:/Users/Aritri Baidya/Desktop/ML Project/Controls_2"

dir.create(output_folder_asd, showWarnings = FALSE)
dir.create(output_folder_ctrl, showWarnings = FALSE)

# === Load Annotation Data ===
annot_df <- fread(annot_path, sep = "\t", header = TRUE, skip = "#", data.table = FALSE)
annot_df <- annot_df[, c("ID", "Symbol")]

# === Load GSM Diagnosis Mapping CSV ===
mapping <- fread("C:/Users/Aritri Baidya/Desktop/ML Project/GSM_diagnosis.csv", data.table = FALSE)

# === Process Each GSM based on Diagnosis ===
for (i in 1:nrow(mapping)) {
  gsm_id <- mapping$GSM_ID[i]
  diagnosis <- mapping$Diagnosis[i]
  
  # Construct filename (assuming standard format: GSMxxxxxxx-tbl-1.txt)
  filename <- paste0(gsm_id, "-tbl-1.txt")
  input_path <- file.path(input_folder, filename)
  
  # Decide output folder
  if (diagnosis == "ASD") {
    output_path <- file.path(output_folder_asd, paste0("expression_data_", gsm_id, ".csv"))
  } else if (diagnosis == "Control") {
    output_path <- file.path(output_folder_ctrl, paste0("expression_data_", gsm_id, ".csv"))
  } else {
    next   # skip unknown diagnosis
  }
  
  # Load expression data
  expr_df <- fread(input_path, sep = "\t", header = FALSE, data.table = FALSE)
  colnames(expr_df) <- c("Probe_ID", "Expression")
  
  # Merge with annotation
  merged_df <- merge(expr_df, annot_df, by.x = "Probe_ID", by.y = "ID", all.x = TRUE)
  merged_df <- merged_df[, c("Probe_ID", "Symbol", "Expression")]
  
  # Save output
  fwrite(merged_df, output_path)
  cat("Saved:", output_path, "\n")
}
