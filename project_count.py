# import pandas as pd
#
# # Read the CSV file into a DataFrame
# df = pd.read_csv('C:/Users/Aritri Baidya/Desktop/ML Project/DEG_results_with_limma_reordered.csv')
#
# # Calculate and print the range for each column
# columns_to_check = ['Log2FC', 'p_value', 'adj_pval']
#
# for col in columns_to_check:
#     min_val = df[col].min()
#     max_val = df[col].max()
#     print(f"Range for {col}: {min_val:.6f} to {max_val:.6f}")
#


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import quantile_transform, StandardScaler
import matplotlib.pyplot as plt
from xgboost import plot_importance
import shap

# --- Step 1: Load data ---
file_path = "C:/Users/Aritri Baidya/Desktop/ML Project/R_codes/ML_ready_dataset_filtered.csv"
df = pd.read_csv(file_path)

# Keep labels separately
labels = df["Condition"].str.strip().str.lower()
samples = df["Sample"]

# Extract only numeric features (gene expression columns)
features = df.drop(columns=["Sample", "Condition"])

# Quantile normalization on features only
features_qn = pd.DataFrame(
    quantile_transform(features, axis=0, n_quantiles=100, output_distribution='normal', copy=True),
    index=features.index, columns=features.columns
)

# Z-score normalization
scaler = StandardScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(features_qn),
    index=features_qn.index, columns=features_qn.columns
)

# Final features (X) and labels (y)
X = features_scaled
y = labels.map({"control": 0, "asd": 1})  # This already encodes to 0,1

print("X shape:", X.shape)
print("y distribution:\n", y.value_counts())

# Check for any missing encodings
print("Unique values in Condition:", df["Condition"].unique())
print("Number of NaN in y:", y.isna().sum())

# Count classes
neg, pos = np.bincount(y)  # y = labels (0=Control, 1=ASD)
print("Controls:", neg, " | ASD:", pos)

# 2. Split your data FIRST to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Perform Univariate Feature Selection on the TRAINING SET only
selector = SelectKBest(score_func=f_classif, k=50)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 4. Check which features were selected
selected_mask = selector.get_support()
feature_names = X.columns
selected_feature_names = feature_names[selected_mask]

print(f"Selected {len(selected_feature_names)} features:")
print(selected_feature_names.tolist())

# 5. Train your model on the selected features
model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train_selected, y_train)

# 6. Evaluate on the test set
y_pred = model.predict(X_test_selected)
y_prob = model.predict_proba(X_test_selected)[:, 1]  # Use probabilities for ROC curve

print("\nClassification Report (with top 50 features):")
print(classification_report(y_test, y_pred, target_names=['Control', 'ASD']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# --- Compute ROC curve and AUC ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Changed y_pred_proba to y_prob
roc_auc = auc(fpr, tpr)

# --- Plot ROC Curve ---
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Model")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
# plt.show()

# --- Feature importance ---
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=20)
plt.title("Top 20 Important Genes (XGBoost)")
plt.tight_layout()
# plt.show()

# --- SHAP values for interpretability ---
# Create DataFrame with selected feature names for SHAP
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_selected_df)

# SHAP summary plot (global feature importance)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train_selected_df, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
# plt.show()

# SHAP beeswarm plot (distribution of impact)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train_selected_df, show=False)
plt.title("SHAP Value Distribution")
plt.tight_layout()
plt.show()

# Dependence plot for the top important gene
if len(selected_feature_names) > 0:
    # Get the top feature from SHAP importance
    shap_df = pd.DataFrame({
        'feature': selected_feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    })
    top_gene = shap_df.sort_values('mean_abs_shap', ascending=False).iloc[0]['feature']

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_gene, shap_values, X_train_selected_df, show=False)
    plt.title(f"SHAP Dependence Plot for {top_gene}")
    plt.tight_layout()
    plt.show()

# # Optional: Get feature importances as DataFrame
# feature_importances = pd.DataFrame({
#     'feature': selected_feature_names,
#     'importance': model.feature_importances_
# }).sort_values('importance', ascending=False)
#
# print("\nTop 10 most important features:")
# print(feature_importances.head(10))