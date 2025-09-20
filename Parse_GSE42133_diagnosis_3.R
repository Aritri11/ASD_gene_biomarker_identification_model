library(xml2)
library(dplyr)

parse_geo_xml_xml2 <- function(xml_file) {
  #"""
  #Parse GEO XML file using xml2 package
  #"""
  # Read XML file
  doc <- read_xml(xml_file)
  
  # Define namespace
  ns <- xml_ns(doc)
  
  # Extract all samples
  samples <- xml_find_all(doc, "//d1:Sample", ns = ns)
  
  results <- lapply(samples, function(sample) {
    gsm_id <- xml_find_first(sample, "./d1:Accession", ns = ns) %>% xml_text()
    
    diagnosis <- xml_find_first(sample, './/d1:Characteristics[@tag="dx (diagnosis)"]', ns = ns) %>% 
      xml_text() %>% 
      trimws()
    
    if (!is.na(diagnosis) && diagnosis != "") {
      data.frame(GSM_ID = gsm_id, Diagnosis = diagnosis, stringsAsFactors = FALSE)
    } else {
      NULL
    }
  })
  
  # Remove NULL entries and combine
  results <- do.call(rbind, results[!sapply(results, is.null)])
  
  return(results)
}

# Usage
xml_file <- "C:/Users/Aritri Baidya/Desktop/ML Project/GSE42133_family/GSE42133_family.xml"
samples <- parse_geo_xml_xml2(xml_file)

cat(paste("Found", nrow(samples), "samples:\n"))
cat("----------------------------------------\n")
print(samples, row.names = FALSE)

# Count by diagnosis
cat("\nDiagnosis counts:\n")
diagnosis_counts <- samples %>% count(Diagnosis)
print(diagnosis_counts)

# === Save to CSV ===
out_file <- "C:/Users/Aritri Baidya/Desktop/ML Project/GSM_diagnosis.csv"
write.csv(samples, out_file, row.names = FALSE)
cat("✅ Saved results to:", out_file, "\n")
