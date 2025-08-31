########## Gene Expression Classification with XGBoost and SHAP Interpretation ###########
#This code trains an XGBoost classifier on gene expression data to distinguish between classes (ASD vs control).
#It evaluates model performance and applies SHAP to interpret feature contributions.

#Author: Aritri Baidya
#Supervisor: Dr. Shyam Sundar Rajagopalan

# Importing necessary modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Function: Load and preprocess dataset

def load_data(file_path: str):
    """
    Loads gene expression data, performs quantile + z-score normalization,
    and prepares features (X) and labels (y).
    """
    df = pd.read_csv(file_path)

    # Extract labels (Condition column: control=0, asd=1)
    labels = df["Condition"].str.strip().str.lower()
    y = labels.map({"control": 0, "asd": 1})

    # Keep only numeric features (drop metadata columns)
    features = df.drop(columns=["Sample", "Condition"])

    # Quantile normalization
    features_qn = pd.DataFrame(
        quantile_transform(features, axis=0, n_quantiles=100,
                           output_distribution='normal', copy=True),
        index=features.index, columns=features.columns
    )

    # Z-score normalization
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features_qn),
        index=features_qn.index, columns=features_qn.columns
    )

    return features_scaled, y



# Function: Run SHAP analysis

def run_shap_analysis(model, X_train_res, X_test, y_test):
    """
    Runs SHAP analysis on a trained model and generates plots.
    """
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model, X_train_res.sample(50, random_state=42))
    shap_values = explainer.shap_values(X_test)

    # --- Plot A: Mean SHAP importance (bar plot) ---
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

    # --- Plot B: Beeswarm summary plot ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=True)

    # --- Plot C: Waterfall for one negative prediction (Control) ---
    neg_pos = np.where(y_test.values == 0)[0][0]
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[neg_pos],
            base_values=explainer.expected_value,
            data=X_test.iloc[neg_pos],
            feature_names=X_test.columns
        ),
        max_display=15,
        show=False
    )
    #Fixing of margins
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()

    # --- Plot D: Waterfall for one positive prediction (ASD) ---
    pos_pos = np.where(y_test.values == 1)[0][0]
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[pos_pos],
            base_values=explainer.expected_value,
            data=X_test.iloc[pos_pos],
            feature_names=X_test.columns
        ),
        max_display=15,
        show=False
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()

    # --- Plot E: Violin plot (per-feature SHAP distribution) ---
    plt.figure(figsize=(10, 6))
    shap.plots.violin(
        shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_test,
            feature_names=X_test.columns
        ),
        max_display=15,
        show=False
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()

    # --- Plot F: Feature importance bar (signed contributions) ---
    plt.figure(figsize=(10, 6))
    shap.plots.bar(
        shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_test,
            feature_names=X_test.columns
        ),
        max_display=15,
        show=False
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()


# Main function

def main():
    # --- Load data ---
    file_path = "C:/Users/Aritri Baidya/Desktop/ML Project/Selected_Genes_Expression.csv"
    X, y = load_data(file_path)

    print("X shape:", X.shape)
    print("y distribution:\n", y.value_counts())

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42, shuffle=True
    )

    # --- Handle imbalance with SMOTE ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", y_train.value_counts())
    print("After SMOTE:", y_train_res.value_counts())

    # --- Build XGBoost classifier ---
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    # --- Predictions ---
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # --- Evaluation ---
    print("Classification Report (test set):")
    print(classification_report(y_test, y_pred, target_names=['Control', 'ASD']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC score: {roc_auc: .2f}")

    # --- ROC curve plot ---
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Final XGBoost Model")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    # --- Run SHAP analysis ---
    run_shap_analysis(model, X_train_res, X_test, y_test)


if __name__ == "__main__":
    main()


# Biological Interpretation:
# All the SHAP plots indicate that ZNF609, NUDT2, and GSTM2 are the strongest predictive genes in the dataset for classification
# So the model predicts these genes to be the most probable gene biomarkers for ASD
