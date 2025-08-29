
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Load dataset
# ================================
file_path = "/home/ibab/Downloads/ML_ready_dataset_filtered.csv"  # adjust if needed
df = pd.read_csv(file_path)

# ================================
# Prepare features (X) and target (y)
# ================================
X = df.drop(columns=["Sample", "Condition"])
y = df["Condition"].map({"ASD": 1, "Control": 0})

# ================================
# Define SVM model
# ================================
svm_model = SVC(kernel="linear", C=1)

# ================================
# RFECV with CV
# ================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rfecv = RFECV(
    estimator=svm_model,
    step=1,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
rfecv.fit(X, y)

# ================================
# Results
# ================================
print(f"Optimal number of features: {rfecv.n_features_}")
selected_features = X.columns[rfecv.support_]
print("Selected Features:")
print(selected_features.tolist())

# # Save ranking
# ranking = pd.DataFrame({
#     "Feature": X.columns,
#     "Rank": rfecv.ranking_,
#     "Selected": rfecv.support_
# }).sort_values("Rank")
# ranking.to_csv("SVM_RFECV_feature_ranking.csv", index=False)
# print("\nFeature ranking saved as 'SVM_RFECV_feature_ranking.csv'")

# ================================
# Plot CV accuracy vs number of features
# ================================
plt.figure(figsize=(8, 6))

# Newer sklearn uses cv_results_ dictionary
if hasattr(rfecv, "cv_results_"):
    scores = rfecv.cv_results_["mean_test_score"]
else:  # fallback for older sklearn
    scores = rfecv.cv_scores_

plt.plot(range(1, len(scores) + 1), scores, marker="o")
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation accuracy")
plt.title("SVM-RFECV Feature Selection")
plt.grid(True)
plt.show()



# ================================
# Create new dataset with only selected genes
# ================================
df_selected = df[["Sample"] + selected_features.tolist() + ["Condition"]]

# Save dataset
# df_selected.to_csv("Selected_Genes_Expression.csv", index=False)
# print("New dataset with selected genes saved as 'Selected_Genes_Expression.csv'")


# ================================
# Heatmap of selected genes
# ================================
# Pivot to matrix (samples × genes)
data_matrix = df_selected.set_index("Sample")
conditions = data_matrix["Condition"]
data_matrix = data_matrix.drop(columns="Condition")

# Create color labels for ASD vs Control
condition_colors = conditions.map({"ASD": "red", "Control": "blue"})

# Plot clustermap
sns.clustermap(
    data_matrix,
    cmap="vlag",
    standard_scale=1,     # normalize each gene
    row_colors=condition_colors,
    figsize=(14, 10)
)

plt.suptitle("Heatmap of Selected Genes (SVM-RFECV)", y=1.02)
plt.savefig("Selected_Genes_Heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("Heatmap saved as 'Selected_Genes_Heatmap.png'")