import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, f1_score, make_scorer
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# =========================================================
# Function: Load and preprocess dataset
# =========================================================
def load_data(file_path: str):
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


# =========================================================
# Function: Hyperparameter tuning
# =========================================================
def tune_hyperparameters(X_train, y_train):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Pipeline with SMOTE + XGBoost
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
    ])

    # Hyperparameter grid
    param_grid = {
        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [3, 4, 5],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__subsample': [0.7, 0.8, 1.0],
        'xgb__colsample_bytree': [0.7, 0.8, 1.0]
    }

    scorer = make_scorer(f1_score, average='weighted')

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring=scorer,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("Best parameters found:")
    print(search.best_params_)
    print("Best weighted F1 score (CV):", search.best_score_)

    return search.best_params_


# =========================================================
# Function: Run SHAP analysis
# =========================================================
def run_shap_analysis(model, X_train_res, X_test, y_test):
    explainer = shap.TreeExplainer(model, X_train_res.sample(50, random_state=42))
    shap_values = explainer.shap_values(X_test)

    # --- Plot A: Mean SHAP importance ---
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

    # --- Plot B: Beeswarm ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=True)

    # --- Waterfall (Control example) ---
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
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()


# =========================================================
# Main workflow
# =========================================================
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

    # --- Hyperparameter tuning ---
    best_params = tune_hyperparameters(X_train, y_train)

    # --- Final model with best params ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=best_params['xgb__n_estimators'],
        max_depth=best_params['xgb__max_depth'],
        learning_rate=best_params['xgb__learning_rate'],
        subsample=best_params['xgb__subsample'],
        colsample_bytree=best_params['xgb__colsample_bytree'],
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
    print(f"ROC-AUC: {roc_auc: .2f}")

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
