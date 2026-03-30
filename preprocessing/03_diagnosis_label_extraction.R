# ============================================================
# STEP 3 — Extract Diagnosis Labels from GEO XML
# Key Features: Label standardization, validation, edge case handling
# ============================================================

library(xml2)
library(dplyr)

# ----------------------------
# 3.1 Path — UPDATE THIS
# ----------------------------
xml_file <- "C:/Users/Aritri Baidya/Desktop/ML Project/GSE42133_family/GSE42133_family.xml"
output_csv <- "C:/Users/Aritri Baidya/Desktop/ML_R_codes/GSM_diagnosis.csv"

# Expected total samples
EXPECTED_SAMPLES <- 147
EXPECTED_ASD     <- 91    # based on published paper
EXPECTED_CTRL    <- 56    # based on published paper

# ----------------------------
# 3.2 Parse GEO XML File
# ----------------------------
cat("Parsing GEO XML file...\n")

parse_geo_xml <- function(xml_file) {

  if (!file.exists(xml_file)) {
    stop(paste("❌ XML file not found:", xml_file))
  }

  doc <- read_xml(xml_file)
  ns  <- xml_ns(doc)

  # Extract all sample nodes
  samples <- xml_find_all(doc, "//d1:Sample", ns = ns)
  cat("Total sample nodes found in XML:", length(samples), "\n")

  results <- lapply(samples, function(sample) {

    # Extract GSM accession ID
    gsm_id <- xml_find_first(sample, "./d1:Accession", ns = ns) %>%
      xml_text() %>%
      trimws()

    # Extract diagnosis — tag name is "dx (diagnosis)"
    diagnosis <- xml_find_first(
      sample,
      './/d1:Characteristics[@tag="dx (diagnosis)"]',
      ns = ns
    ) %>%
      xml_text() %>%
      trimws()

    # Return row if diagnosis found
    if (!is.na(diagnosis) && diagnosis != "") {
      data.frame(GSM_ID    = gsm_id,
                 Diagnosis = diagnosis,
                 stringsAsFactors = FALSE)
    } else {
      # Try alternative tag names
      alt_diag <- xml_find_first(
        sample,
        './/d1:Characteristics[@tag="diagnosis"]',
        ns = ns
      ) %>% xml_text() %>% trimws()

      if (!is.na(alt_diag) && alt_diag != "") {
        data.frame(GSM_ID    = gsm_id,
                   Diagnosis = alt_diag,
                   stringsAsFactors = FALSE)
      } else {
        cat("⚠️  No diagnosis found for:", gsm_id, "\n")
        return(NULL)
      }
    }
  })

  # Remove NULLs and combine
  results <- do.call(rbind, results[!sapply(results, is.null)])
  return(results)
}

samples <- parse_geo_xml(xml_file)
cat("✅ Diagnosis extracted for", nrow(samples), "samples.\n\n")

# ----------------------------
# 3.3 Standardize Diagnosis Labels
# ----------------------------
cat("=== RAW DIAGNOSIS VALUES ===\n")
print(table(samples$Diagnosis))
cat("\n")

# Standardize to "ASD" and "Control"
samples <- samples %>%
  mutate(Diagnosis = trimws(Diagnosis),
         Diagnosis = case_when(
           Diagnosis %in% c("ASD", "Autism", "autism",
                             "Autistic", "AUTISM")          ~ "ASD",
           Diagnosis %in% c("Control", "control", "TD",
                             "Typically Developing",
                             "typically developing",
                             "CONTROL", "Normal")            ~ "Control",
           TRUE                                              ~ Diagnosis  # keep unknown as-is
         ))

cat("=== STANDARDIZED DIAGNOSIS VALUES ===\n")
print(table(samples$Diagnosis))
cat("\n")

# ----------------------------
# 3.4 Validation Checks
# ----------------------------
cat("=== VALIDATION ===\n")

# Check total count
if (nrow(samples) == EXPECTED_SAMPLES) {
  cat("✅ Total samples:", nrow(samples), "(matches expected", EXPECTED_SAMPLES, ")\n")
} else {
  cat("⚠️  Total samples:", nrow(samples),
      "— expected", EXPECTED_SAMPLES, "\n")
}

# Check ASD count
n_asd <- sum(samples$Diagnosis == "ASD")
if (n_asd == EXPECTED_ASD) {
  cat("✅ ASD samples:", n_asd, "(matches expected", EXPECTED_ASD, ")\n")
} else {
  cat("⚠️  ASD samples:", n_asd, "— expected", EXPECTED_ASD, "\n")
}

# Check Control count
n_ctrl <- sum(samples$Diagnosis == "Control")
if (n_ctrl == EXPECTED_CTRL) {
  cat("✅ Control samples:", n_ctrl, "(matches expected", EXPECTED_CTRL, ")\n")
} else {
  cat("⚠️  Control samples:", n_ctrl, "— expected", EXPECTED_CTRL, "\n")
}

# Check for unexpected labels
unexpected <- samples$Diagnosis[!samples$Diagnosis %in% c("ASD", "Control")]
if (length(unexpected) > 0) {
  cat("\n⚠️  Unexpected diagnosis labels found:\n")
  print(unique(unexpected))
  cat("These samples will be excluded from analysis.\n")

  # Filter to only valid labels
  samples <- samples %>%
    filter(Diagnosis %in% c("ASD", "Control"))
  cat("Samples after filtering:", nrow(samples), "\n")
} else {
  cat("✅ All diagnosis labels are valid (ASD or Control).\n")
}

# Check for duplicate GSM IDs
dup_gsm <- samples$GSM_ID[duplicated(samples$GSM_ID)]
if (length(dup_gsm) > 0) {
  cat("\n⚠️  Duplicate GSM IDs found:", dup_gsm, "\n")
  samples <- samples[!duplicated(samples$GSM_ID), ]
  cat("Duplicates removed. Final count:", nrow(samples), "\n")
} else {
  cat("✅ No duplicate GSM IDs found.\n")
}

# ----------------------------
# 3.5 Save Output
# ----------------------------
write.csv(samples, output_csv, row.names = FALSE)
cat("\n✅ Diagnosis mapping saved:", output_csv, "\n")

cat("\n=== FINAL DIAGNOSIS SUMMARY ===\n")
print(table(samples$Diagnosis))

cat("\n========================================\n")
cat("STEP 3 COMPLETE\n")
cat("Output: GSM_diagnosis.csv\n")
cat("  ASD samples:", sum(samples$Diagnosis == "ASD"), "\n")
cat("  Control samples:", sum(samples$Diagnosis == "Control"), "\n")
cat("========================================\n")
