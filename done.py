import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBClassifier
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# np.random.default_rng(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def load_data(file_path):
    df = pd.read_csv(file_path)
    features = df.drop(columns=["Sample", "Condition"])
    print("No of features: ", features.shape[1])

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
    X = features_scaled
    y = df["Condition"].map({"ASD": 1, "Control": 0})
    return X, y, df


def run_shap_analysis(model, X_train_res, X_test, y_test):
    """
    Runs SHAP analysis on a trained model and generates plots.
    """

    explainer = shap.TreeExplainer(model, X_train_res.sample(50, random_state=42))
    shap_values = explainer.shap_values(X_test)

    # --- SHAP summary plots ---
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)
    shap.summary_plot(shap_values, X_test, show=True)

    # --- Waterfall plot (Control) ---
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
    # Fixing of margins
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()

    # --- Waterfall plot (ASD) ---
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
    # Fixing of margins
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()


    # --- Violin plot ---
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
    # Fixing of margins
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()


    # --- Feature importance bar ---
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
    # Fixing of margins
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()



def main():
    # Step 1: Load dataset
    file_path = "ML_ready_dataset_filtered.csv"
    X, y, df = load_data(file_path)

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    # Step 2b: Variance threshold
    vt = VarianceThreshold(threshold=0.7)
    X_train_vt = vt.fit_transform(X_train)
    X_test_vt = vt.transform(X_test)
    print(f"Features after VarianceThreshold: {X_train_vt.shape[1]}")
    selected_vt_features = X_train.columns[vt.get_support()]

    # Step 3: Handle imbalance
    smote = SMOTE(sampling_strategy={0: 51, 1: 63}, k_neighbors=8, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_vt, y_train)
    print("Before SMOTE:", y_train.value_counts())
    print("After SMOTE:", y_train_res.value_counts())

    # Step 4: Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=80)
    X_train_selected = selector.fit_transform(X_train_res, y_train_res)
    X_test_selected = selector.transform(X_test_vt)

    selected_features = selected_vt_features[selector.get_support()]
    print(f"Number of features after selection: {X_train_selected.shape[1]}")

    # Step 5: Hyperopt space
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'gamma': hp.uniform('gamma', 0, 5),
    }

    def objective(params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])

        clf = XGBClassifier(**params, random_state=42, eval_metric="logloss")
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in folds.split(X_train_selected, y_train_res):
            x_tr, x_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
            y_tr, y_val_fold = y_train_res[train_idx], y_train_res[val_idx]

            clf.fit(x_tr, y_tr)
            y_pred = clf.predict(x_val_fold)
            f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))

        return {'loss': -np.mean(f1_scores), 'status': STATUS_OK, 'params': params}

    # Step 6: Hyperopt
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=30, trials=trials, rstate=np.random.default_rng(42))
    best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
    best_params = best_trial['params']

    # Step 7: Final model
    clf = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
    clf.fit(X_train_selected, y_train_res)

    # Step 8: Evaluation
    print("\nBest XGBoost Params:", best_params)
    y_pred_train=clf.predict(X_train_selected)
    y_pred_test=clf.predict(X_test_selected)
    print(f"Train Accuracy: {accuracy_score(y_train_res,y_pred_train )*100: .2f}%")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)*100: .2f}%")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, clf.predict(X_test_selected), target_names=["Control", "ASD"]))
    print(confusion_matrix(y_test, clf.predict(X_test_selected)))

    # --- ROC-AUC ---
    y_prob = clf.predict_proba(X_test_selected)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC score: {roc_auc:.2f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Final XGBoost Model")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    # --- Run SHAP analysis ---
    run_shap_analysis(clf, pd.DataFrame(X_train_selected, columns=selected_features),
                      pd.DataFrame(X_test_selected, columns=selected_features), y_test)


if __name__ == "__main__":
    main()
