library(GEOquery)

# --- Load the GEO series matrix file directly from NCBI ---

# This automatically downloads and parses the GSE42133 dataset
gse <- getGEO("GSE42133", GSEMatrix = TRUE)[[1]]

# --- View basic metadata ---
print(gse)

# --- Extract expression matrix ---
expr <- exprs(gse)   # rows = genes/probes, cols = samples
print(expr)

summary(as.vector(expr))
hist(as.vector(expr), breaks = 100, main = "Distribution of Expression Values", xlab = "Expression")


# --- Extract sample metadata ---
meta <- pData(gse)   # phenotype data for each sample
print(meta)

# --- View selected metadata columns ---
head(meta[, c("title", "source_name_ch1", "characteristics_ch1.1")])

