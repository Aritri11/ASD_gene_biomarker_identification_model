# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler, quantile_transform
# from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
# from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE
#
# # --- Load data ---
# file_path = "/home/ibab/ML_project/Selected_Genes_Expression.csv"
# df = pd.read_csv(file_path)
#
# # Labels
# labels = df["Condition"].str.strip().str.lower()
# X = df.drop(columns=["Sample", "Condition"])
# y = labels.map({"control": 0, "asd": 1})
#
# # Quantile normalization + Z-score
# X_qn = pd.DataFrame(
#     quantile_transform(X, axis=0, n_quantiles=100, output_distribution='normal', copy=True),
#     index=X.index, columns=X.columns
# )
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X_qn), index=X_qn.index, columns=X_qn.columns)
#
# # Stratified K-Fold CV
# n_splits = 5
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#
# # Track metrics
# f1_scores = []
# roc_aucs = []
# best_thresholds = []
#
# fold = 1
# for train_idx, test_idx in skf.split(X_scaled, y):
#     X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#
#     # Apply SMOTE on training fold only
#     smote = SMOTE(random_state=42)
#     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#
#     # Train XGBoost
#     model = XGBClassifier(random_state=42, eval_metric='logloss')
#     model.fit(X_train_res, y_train_res)
#
#     # Predict probabilities
#     y_prob = model.predict_proba(X_test)[:, 1]
#
#     # Find best threshold based on weighted F1
#     thresholds = np.arange(0.3, 0.71, 0.05)
#     best_f1 = 0
#     best_thresh = 0.5
#     for t in thresholds:
#         y_pred_t = (y_prob >= t).astype(int)
#         f1 = f1_score(y_test, y_pred_t, average='weighted')
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thresh = t
#
#     best_thresholds.append(best_thresh)
#
#     # Final predictions with best threshold
#     y_pred_final = (y_prob >= best_thresh).astype(int)
#
#     # Metrics
#     f1_scores.append(f1_score(y_test, y_pred_final, average='weighted'))
#     roc_aucs.append(roc_auc_score(y_test, y_prob))
#
#     print(f"\n--- Fold {fold} ---")
#     print(f"Best threshold: {best_thresh:.2f}")
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred_final, target_names=['Control', 'ASD']))
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred_final))
#
#     fold += 1
#
# print("\n=== CROSS-VALIDATION RESULTS ===")
# print(f"Average weighted F1-score: {np.mean(f1_scores):.3f}")
# print(f"Average ROC-AUC: {np.mean(roc_aucs):.3f}")
# print(f"Average best threshold: {np.mean(best_thresholds):.2f}")






























# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, quantile_transform
# from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
# from sklearn.metrics import make_scorer, f1_score
# from xgboost import XGBClassifier
# from imblearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
#
# # --- Load data ---
# file_path = "/home/ibab/ML_project/Selected_Genes_Expression.csv"
# df = pd.read_csv(file_path)
#
# # Features and labels
# labels = df["Condition"].str.strip().str.lower()
# X = df.drop(columns=["Sample", "Condition"])
# y = labels.map({"control": 0, "asd": 1})
#
# # Quantile normalization + Z-score
# X_qn = pd.DataFrame(
#     quantile_transform(X, axis=0, n_quantiles=100, output_distribution='normal', copy=True),
#     index=X.index, columns=X.columns
# )
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X_qn), index=X_qn.index, columns=X_qn.columns)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.3, stratify=y, random_state=42
# )
#
#
#
# # --- Stratified K-Fold ---
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#
# # --- Pipeline: SMOTE + XGBoost ---
# pipeline = Pipeline([
#     ('smote', SMOTE(random_state=42)),
#     ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
# ])
#
# # --- Hyperparameter grid ---
# param_grid = {
#     'xgb__n_estimators': [100, 200, 300],
#     'xgb__max_depth': [3, 4, 5],
#     'xgb__learning_rate': [0.01, 0.05, 0.1],
#     'xgb__subsample': [0.7, 0.8, 1.0],
#     'xgb__colsample_bytree': [0.7, 0.8, 1.0]
# }
#
# # --- Scorer ---
# scorer = make_scorer(f1_score, average='weighted')
#
# # --- Randomized search ---
# search = RandomizedSearchCV(
#     estimator=pipeline,
#     param_distributions=param_grid,
#     n_iter=20,           # number of random combinations
#     scoring=scorer,
#     cv=cv,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )
#
# # --- Fit search ---
# search.fit(X_train, y_train)
#
# # --- Best parameters ---
# print("Best parameters found:")
# print(search.best_params_)
# print("Best weighted F1 score (CV):", search.best_score_)
#
#
# # Extract best parameters
# best_params = search.best_params_
#
# # Build final pipeline
# final_pipeline = Pipeline([
#     ('smote', SMOTE(random_state=42)),
#     ('xgb', XGBClassifier(
#         n_estimators=best_params['xgb__n_estimators'],
#         max_depth=best_params['xgb__max_depth'],
#         learning_rate=best_params['xgb__learning_rate'],
#         subsample=best_params['xgb__subsample'],
#         colsample_bytree=best_params['xgb__colsample_bytree'],
#         eval_metric='logloss',
#         random_state=42
#     ))
# ])
#
# # Train on full dataset
# final_pipeline.fit(X_train, y_train)
# print("Final model trained on all data with optimal parameters.")






























import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from xgboost import plot_importance
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# --- Load data ---
file_path = "/home/ibab/ML_project/Selected_Genes_Expression.csv"
df = pd.read_csv(file_path)

# Features and labels
labels = df["Condition"].str.strip().str.lower()
X = df.drop(columns=["Sample", "Condition"])
y = labels.map({"control": 0, "asd": 1})

# Quantile normalization + Z-score
X_qn = pd.DataFrame(
    quantile_transform(X, axis=0, n_quantiles=100, output_distribution='normal', copy=True),
    index=X.index, columns=X.columns
)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_qn), index=X_qn.index, columns=X_qn.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=42, shuffle=True
)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- Build final robust model (train only on training data) ---
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

# Train on SMOTE-resampled training data
model.fit(X_train_res, y_train_res)

# --- Predictions on untouched test set ---
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# --- Classification metrics ---
print("Classification Report (test set):")
print(classification_report(y_test, y_pred, target_names=['Control', 'ASD']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# --- ROC curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Final XGBoost Model")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# --- Feature importance ---
plt.figure(figsize=(12, 8))
plot_importance(model, max_num_features=25)
plt.title("Top 25 Important Genes (XGBoost)")
plt.tight_layout()
plt.show()

# --- SHAP values ---
explainer = shap.TreeExplainer(model,X_train_res.sample(50, random_state=42))
shap_values = explainer.shap_values(X_test)  # evaluate SHAP on test data

# SHAP summary plot (bar)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# SHAP beeswarm plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=True)

# Dependence plot for top gene
shap_df = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
})
top_gene = shap_df.sort_values('mean_abs_shap', ascending=False).iloc[0]['feature']

plt.figure(figsize=(10, 6))
shap.dependence_plot(top_gene, shap_values, X_test)
plt.title(f"SHAP Dependence Plot for {top_gene}")
plt.show()


