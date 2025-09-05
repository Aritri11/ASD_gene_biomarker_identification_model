# Load required libraries
library(tidyverse)
library(limma)

# ----------------------------
# Step 1: Load expression data
# ----------------------------
file_path <- "C:/Users/Aritri Baidya/Desktop/ML Project/merged_data_long_format.csv"
df <- read_csv(file_path)

# Pivot to wide format (genes x samples)
df_expr <- df %>%
  filter(!is.na(Symbol)) %>%
  group_by(Symbol, Sample) %>%
  summarise(Expression = mean(Expression, na.rm = TRUE), .groups = 'drop') %>%
  pivot_wider(names_from = Sample, values_from = Expression) %>%
  column_to_rownames("Symbol")

# Extract conditions
conditions <- df %>%
  distinct(Sample, Condition) %>%
  deframe()

df_expr <- df_expr[, names(conditions)]  # ensure order matches

# ----------------------------
# Step 2: DEG analysis (limma)
# ----------------------------
colData <- data.frame(Sample = names(conditions),
                      Condition = factor(conditions, levels = c("Control", "ASD")))  # force Control baseline

design <- model.matrix(~ Condition, data = colData)
colnames(design)  # should now be (Intercept), ConditionASD

fit <- lmFit(df_expr, design)
fit <- eBayes(fit)

limma_results <- topTable(fit, coef = "ConditionASD", number = Inf, sort.by = "none") %>%
  rownames_to_column("Gene") %>%
  select(Gene, logFC, P.Value, adj.P.Val) %>%
  rename(Log2FC = logFC, p_value = P.Value, adj_pval = adj.P.Val)

# # Save full DEG results
# write_csv(limma_results, "/home/ibab/PythonProject/ML_Project/DEG_results_with_limma.csv")

# ----------------------------
# Step 3: Volcano plot
# ----------------------------
volcano_data <- limma_results %>%
  mutate(
    status = case_when(
      Log2FC >= 0.5 & adj_pval < 0.05 ~ "Upregulated",
      Log2FC <= -0.5 & adj_pval < 0.05 ~ "Downregulated",
      TRUE ~ "Not Significant"
    )
  )

p <- ggplot(volcano_data, aes(x = Log2FC, y = -log10(adj_pval), color = status)) +
  geom_point(alpha = 0.7, size = 1.5) +
  geom_hline(yintercept = -log10(0.05), color = "black", linetype = "dashed") +
  geom_vline(xintercept = c(-1, 1), color = "black", linetype = "dashed", alpha = 0.5) +
  scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "gray")) +
  labs(title = "Volcano Plot (limma)",
       x = "Log2 Fold Change",
       y = "-log10(Adjusted P-value)",
       color = "Gene Regulation") +
  theme_minimal() +
  theme(legend.position = "right")
ggsave("/home/ibab/PythonProject/ML_Project/volcano_plot_limma.png", plot = p, width = 8, height = 6)

# ----------------------------
# Step 4: Filter significant DEGs
# ----------------------------
filtered_genes <- limma_results %>%
  filter(abs(Log2FC) > 0.2, adj_pval < 0.1)

gene_candidates <- unique(filtered_genes$Gene)

expr <- as.matrix(df_expr)
expr_filtered <- expr[rownames(expr) %in% gene_candidates, ]

# ----------------------------
# Step 5: Build ML-ready dataset
# ----------------------------
expr_for_ml <- t(expr_filtered)  # samples × genes
ml_data <- data.frame(Sample = rownames(expr_for_ml), expr_for_ml)  # keep sample names
ml_data$Condition <- colData$Condition[match(ml_data$Sample, colData$Sample)]  # align condition with sample

# Save ML-ready dataset
write_csv(ml_data, "C:/Users/Aritri Baidya/Desktop/ML Project/R_codes/ML_ready_dataset_filtered.csv")

cat("✅ DEG results, volcano plot, and filtered ML dataset saved with Sample column.\n")