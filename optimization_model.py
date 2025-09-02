# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, quantile_transform
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, f1_score, make_scorer
# import matplotlib.pyplot as plt
# import shap
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
#
#
# # =========================================================
# # Function: Load and preprocess dataset
# # =========================================================
# def load_data(file_path: str):
#     df = pd.read_csv(file_path)
#
#     # Extract labels (Condition column: control=0, asd=1)
#     labels = df["Condition"].str.strip().str.lower()
#     y = labels.map({"control": 0, "asd": 1})
#
#     # Keep only numeric features (drop metadata columns)
#     features = df.drop(columns=["Sample", "Condition"])
#
#     # Quantile normalization
#     features_qn = pd.DataFrame(
#         quantile_transform(features, axis=0, n_quantiles=100,
#                            output_distribution='normal', copy=True),
#         index=features.index, columns=features.columns
#     )
#
#     # Z-score normalization
#     scaler = StandardScaler()
#     features_scaled = pd.DataFrame(
#         scaler.fit_transform(features_qn),
#         index=features_qn.index, columns=features_qn.columns
#     )
#
#     return features_scaled, y
#
#
# # =========================================================
# # Function: Hyperparameter tuning
# # =========================================================
# def tune_hyperparameters(X_train, y_train):
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#
#     # Pipeline with SMOTE + XGBoost
#     pipeline = Pipeline([
#         ('smote', SMOTE(random_state=42)),
#         ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
#     ])
#
#     # Hyperparameter grid
#     param_grid = {
#         'xgb__n_estimators': [100, 200, 300],
#         'xgb__max_depth': [3, 4, 5],
#         'xgb__learning_rate': [0.01, 0.05, 0.1],
#         'xgb__subsample': [0.7, 0.8, 1.0],
#         'xgb__colsample_bytree': [0.7, 0.8, 1.0]
#     }
#
#     scorer = make_scorer(f1_score, average='weighted')
#
#     search = RandomizedSearchCV(
#         estimator=pipeline,
#         param_distributions=param_grid,
#         n_iter=20,
#         scoring=scorer,
#         cv=cv,
#         verbose=2,
#         random_state=42,
#         n_jobs=-1
#     )
#
#     search.fit(X_train, y_train)
#
#     print("Best parameters found:")
#     print(search.best_params_)
#     print("Best weighted F1 score (CV):", search.best_score_)
#
#     return search.best_params_
#
#
# # =========================================================
# # Function: Run SHAP analysis
# # =========================================================
# def run_shap_analysis(model, X_train_res, X_test, y_test):
#     explainer = shap.TreeExplainer(model, X_train_res.sample(50, random_state=42))
#     shap_values = explainer.shap_values(X_test)
#
#     # --- Plot A: Mean SHAP importance ---
#     plt.figure(figsize=(8, 6))
#     shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)
#
#     # --- Plot B: Beeswarm ---
#     plt.figure(figsize=(10, 6))
#     shap.summary_plot(shap_values, X_test, show=True)
#
#     # --- Waterfall (Control example) ---
#     neg_pos = np.where(y_test.values == 0)[0][0]
#     shap.waterfall_plot(
#         shap.Explanation(
#             values=shap_values[neg_pos],
#             base_values=explainer.expected_value,
#             data=X_test.iloc[neg_pos],
#             feature_names=X_test.columns
#         ),
#         max_display=15,
#         show=False
#     )
#     fig = plt.gcf()
#     fig.set_size_inches(10, 8)
#     plt.subplots_adjust(left=0.25)
#     plt.show()
#
#
# # =========================================================
# # Main workflow
# # =========================================================
# def main():
#     # --- Load data ---
#     file_path = "C:/Users/Aritri Baidya/Desktop/ML Project/Selected_Genes_Expression.csv"
#     X, y = load_data(file_path)
#
#     print("X shape:", X.shape)
#     print("y distribution:\n", y.value_counts())
#
#     # --- Train-test split ---
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, stratify=y, random_state=42, shuffle=True
#     )
#
#     # --- Hyperparameter tuning ---
#     best_params = tune_hyperparameters(X_train, y_train)
#
#     # --- Final model with best params ---
#     smote = SMOTE(random_state=42)
#     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#
#     model = XGBClassifier(
#         n_estimators=best_params['xgb__n_estimators'],
#         max_depth=best_params['xgb__max_depth'],
#         learning_rate=best_params['xgb__learning_rate'],
#         subsample=best_params['xgb__subsample'],
#         colsample_bytree=best_params['xgb__colsample_bytree'],
#         eval_metric='logloss',
#         random_state=42
#     )
#
#     model.fit(X_train_res, y_train_res)
#
#     # --- Predictions ---
#     y_prob = model.predict_proba(X_test)[:, 1]
#     y_pred = model.predict(X_test)
#
#     # --- Evaluation ---
#     print("Classification Report (test set):")
#     print(classification_report(y_test, y_pred, target_names=['Control', 'ASD']))
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
#     roc_auc = roc_auc_score(y_test, y_prob)
#     print(f"ROC-AUC: {roc_auc: .2f}")
#
#     # --- ROC curve plot ---
#     fpr, tpr, thresholds = roc_curve(y_test, y_prob)
#     roc_auc_val = auc(fpr, tpr)
#
#     plt.figure(figsize=(7, 6))
#     plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
#     plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random guess")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve - Final XGBoost Model")
#     plt.legend(loc="lower right")
#     plt.grid(alpha=0.3)
#     plt.show()
#
#     # --- Run SHAP analysis ---
#     run_shap_analysis(model, X_train_res, X_test, y_test)
#
#
# if __name__ == "__main__":
#     main()
#





################################################################################################################################################################

#SVC

#
# import pandas as pd
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2, mutual_info_classif
# from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler, quantile_transform
# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
#
#
# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     X = df.drop(columns=["Sample", "Condition"])
#     y = df["Condition"].map({"ASD": 1, "Control": 0})
#     return X, y, df
#
#
# def main():
#     # Step 1: Load dataset
#     file_path = "ML_ready_dataset_filtered.csv"
#     X, y, df = load_data(file_path)
#     features = df.drop(columns=["Sample", "Condition"])
#
#     # Quantile normalization
#     features_qn = pd.DataFrame(
#         quantile_transform(features, axis=0, n_quantiles=100,
#                            output_distribution='normal', copy=True),
#         index=features.index, columns=features.columns
#     )
#
#     # Z-score normalization
#     scaler = StandardScaler()
#     features_scaled = pd.DataFrame(
#         scaler.fit_transform(features_qn),
#         index=features_qn.index, columns=features_qn.columns
#     )
#
#     # Step 2: Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, stratify=y, random_state=42
#     )
#
#     # --- Handle imbalance with SMOTE ---
#     smote = SMOTE(sampling_strategy={0: 50, 1: 63}, k_neighbors=8,random_state=42)
#     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#
#     print("Before SMOTE:", y_train.value_counts())
#     # print("After SMOTE:", y_train_res.value_counts())
#
#     # Step 3: Apply VarianceThreshold
#     # varthres = VarianceThreshold(threshold=(.8 * (1 - .8)))
#     # X_train_selected = varthres.fit_transform(X_train_res)
#     # X_test_selected = varthres.transform(X_test)
#     #
#     # selected_features = X.columns[varthres.get_support()]
#
#     selector=SelectKBest(score_func=mutual_info_classif, k=50)
#     X_train_selected = selector.fit_transform(X_train_res,y_train_res)
#     X_test_selected = selector.transform(X_test)
#
#     selected_features = X.columns[selector.get_support()]
#     print(f"Number of features after VarianceThreshold: {X_train_selected.shape[1]}")
#     print("Selected features:", selected_features)
#
#     space = {
#         'C': hp.loguniform('C', -3, 3),  # range ~ [0.05, 20]
#         'gamma': hp.loguniform('gamma', -4, 1),  # range ~ [0.018, 2.7]
#         'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
#         'degree': hp.quniform('degree', 2, 5, 1),  # used if poly
#         'coef0': hp.uniform('coef0', 0, 1)  # for poly/sigmoid
#     }
#
#     def objective(params):
#         params['degree'] = int(params['degree'])
#
#         clf = SVC(**params, probability=True, random_state=42)
#         folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#         f1_scores = []
#
#         for train_idx, val_idx in folds.split(X_train_selected, y_train_res):
#             x_tr, x_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
#             y_tr, y_val_fold = y_train_res[train_idx], y_train_res[val_idx]
#
#             clf.fit(x_tr, y_tr)
#             y_pred = clf.predict(x_val_fold)
#             f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))
#
#         return {'loss': -np.mean(f1_scores), 'status': STATUS_OK, 'params': params}
#
#     trials = Trials()
#     best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30,
#                 trials=trials, rstate=np.random.default_rng(42))
#     print("\nBest Hyperparameters:", best)
#
#     best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
#     best_params = best_trial['params']
#
#     # # Step 6: Train final model with best params
#     clf = SVC(**best_params, probability=True, random_state=42)
#     clf.fit(X_train_selected, y_train_res)
#
#     # Step 7: Evaluate on test set
#     print("\nBest SVC Params:", best_params)
#     print("Train Accuracy:", accuracy_score(y_train_res, clf.predict(X_train_selected)))
#     print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test_selected)))
#     print("\nClassification Report (Test):")
#     print(classification_report(y_test, clf.predict(X_test_selected), target_names=["Control", "ASD"]))
#     print(confusion_matrix(y_test, clf.predict(X_test_selected)))
#
#
# if __name__ == "__main__":
#     main()


####################################################################################################################################

#gradientboost


# import pandas as pd
# import numpy as np
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold, f_classif
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler, quantile_transform
# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
#
#
# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     X = df.drop(columns=["Sample", "Condition"])
#     y = df["Condition"].map({"ASD": 1, "Control": 0})
#     return X, y, df
#
#
# def main():
#     # Step 1: Load dataset
#     file_path = "ML_ready_dataset_filtered.csv"
#     X, y, df = load_data(file_path)
#     features = df.drop(columns=["Sample", "Condition"])
#
#     # Quantile normalization
#     features_qn = pd.DataFrame(
#         quantile_transform(features, axis=0, n_quantiles=100,
#                            output_distribution='normal', copy=True),
#         index=features.index, columns=features.columns
#     )
#
#     # Z-score normalization
#     scaler = StandardScaler()
#     features_scaled = pd.DataFrame(
#         scaler.fit_transform(features_qn),
#         index=features_qn.index, columns=features_qn.columns
#     )
#     X=features_scaled
#     y = df["Condition"].map({"ASD": 1, "Control": 0})
#
#     # Step 2: Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, stratify=y, random_state=42
#     )
#
#     # --- Handle imbalance with SMOTE ---
#     smote = SMOTE(sampling_strategy={0: 50, 1: 63}, k_neighbors=8, random_state=42)
#     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#
#     print("Before SMOTE:", y_train.value_counts())
#
#     # Step 3: Feature selection
#     selector = SelectKBest(score_func=f_classif, k=40)
#     X_train_selected = selector.fit_transform(X_train_res, y_train_res)
#     X_test_selected = selector.transform(X_test)
#
#     selected_features = X.columns[selector.get_support()]
#
#     # selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
#     # X_train_selected = selector.fit_transform(X_train_res, y_train_res)
#     # X_test_selected = selector.transform(X_test)
#     #
#     # selected_features = X.columns[selector.get_support()]
#     print(f"Number of features selected: {X_train_selected.shape[1]}")
#     print("Selected features:", selected_features.tolist())
#
#     # Step 4: Define hyperparameter space for Gradient Boosting
#     space = {
#         'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
#         'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
#         'max_depth': hp.quniform('max_depth', 2, 6, 1),
#         'subsample': hp.uniform('subsample', 0.6, 1.0),
#         'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
#         'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1)
#     }
#
#     def objective(params):
#         params['n_estimators'] = int(params['n_estimators'])
#         params['max_depth'] = int(params['max_depth'])
#         params['min_samples_split'] = int(params['min_samples_split'])
#         params['min_samples_leaf'] = int(params['min_samples_leaf'])
#
#         clf = GradientBoostingClassifier(**params, random_state=42)
#         folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#         f1_scores = []
#
#         for train_idx, val_idx in folds.split(X_train_selected, y_train_res):
#             x_tr, x_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
#             y_tr, y_val_fold = y_train_res[train_idx], y_train_res[val_idx]
#
#             clf.fit(x_tr, y_tr)
#             y_pred = clf.predict(x_val_fold)
#             f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))
#
#         return {'loss': -np.mean(f1_scores), 'status': STATUS_OK, 'params': params}
#
#     # Step 5: Run hyperopt optimization
#     trials = Trials()
#     best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30,
#                 trials=trials, rstate=np.random.default_rng(42))
#     print("\nBest Hyperparameters:", best)
#
#     best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
#     best_params = best_trial['params']
#
#     # Step 6: Train final model with best params
#     clf = GradientBoostingClassifier(**best_params, random_state=42)
#     clf.fit(X_train_selected, y_train_res)
#
#     # Step 7: Evaluate on test set
#     print("\nBest Gradient Boosting Params:", best_params)
#     print("Train Accuracy:", accuracy_score(y_train_res, clf.predict(X_train_selected)))
#     print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test_selected)))
#     print("\nClassification Report (Test):")
#     print(classification_report(y_test, clf.predict(X_test_selected), target_names=["Control", "ASD"]))
#     print(confusion_matrix(y_test, clf.predict(X_test_selected)))
#
#
# if __name__ == "__main__":
#     main()



#########################################################################################################################################

#RandomForest

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler, quantile_transform
# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
#
#
# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     X = df.drop(columns=["Sample", "Condition"])
#     y = df["Condition"].map({"ASD": 1, "Control": 0})
#     return X, y, df
#
#
# def main():
#     # Step 1: Load dataset
#     file_path = "ML_ready_dataset_filtered.csv"
#     X, y, df = load_data(file_path)
#     features = df.drop(columns=["Sample", "Condition"])
#
#     # Quantile normalization
#     features_qn = pd.DataFrame(
#         quantile_transform(features, axis=0, n_quantiles=100,
#                            output_distribution='normal', copy=True),
#         index=features.index, columns=features.columns
#     )
#
#     # Z-score normalization
#     scaler = StandardScaler()
#     features_scaled = pd.DataFrame(
#         scaler.fit_transform(features_qn),
#         index=features_qn.index, columns=features_qn.columns
#     )
#
#     X = features_scaled
#     y = df["Condition"].map({"ASD": 1, "Control": 0})
#
#     # Step 2: Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, stratify=y, random_state=42
#     )
#
#     # --- Handle imbalance with SMOTE ---
#     smote = SMOTE(sampling_strategy={0: 50, 1: 63}, k_neighbors=8, random_state=42)
#     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#
#     print("Before SMOTE:", y_train.value_counts())
#     print("After SMOTE:", y_train_res.value_counts())
#
#     # Step 3: Feature selection
#     selector = SelectKBest(score_func=mutual_info_classif, k=30)
#     X_train_selected = selector.fit_transform(X_train_res, y_train_res)
#     X_test_selected = selector.transform(X_test)
#
#     selected_features = X.columns[selector.get_support()]
#     print(f"Number of features selected: {X_train_selected.shape[1]}")
#     print("Selected features:", selected_features.tolist())
#
#     # Step 4: Define hyperparameter space for Random Forest
#     space = {
#         'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
#         'max_depth': hp.quniform('max_depth', 3, 15, 1),
#         'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
#         'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
#         'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
#         'bootstrap': hp.choice('bootstrap', [True, False])
#     }
#
#     def objective(params):
#         params['n_estimators'] = int(params['n_estimators'])
#         params['max_depth'] = int(params['max_depth'])
#         params['min_samples_split'] = int(params['min_samples_split'])
#         params['min_samples_leaf'] = int(params['min_samples_leaf'])
#
#         clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
#         folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#         f1_scores = []
#
#         for train_idx, val_idx in folds.split(X_train_selected, y_train_res):
#             x_tr, x_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
#             y_tr, y_val_fold = y_train_res[train_idx], y_train_res[val_idx]
#
#             clf.fit(x_tr, y_tr)
#             y_pred = clf.predict(x_val_fold)
#             f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))
#
#         return {'loss': -np.mean(f1_scores), 'status': STATUS_OK, 'params': params}
#
#     # Step 5: Run hyperopt optimization
#     trials = Trials()
#     best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30,
#                 trials=trials, rstate=np.random.default_rng(42))
#     print("\nBest Hyperparameters:", best)
#
#     best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
#     best_params = best_trial['params']
#
#     # Step 6: Train final model with best params
#     clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
#     clf.fit(X_train_selected, y_train_res)
#
#     # Step 7: Evaluate on test set
#     print("\nBest Random Forest Params:", best_params)
#     print("Train Accuracy:", accuracy_score(y_train_res, clf.predict(X_train_selected)))
#     print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test_selected)))
#     print("\nClassification Report (Test):")
#     print(classification_report(y_test, clf.predict(X_test_selected), target_names=["Control", "ASD"]))
#     print(confusion_matrix(y_test, clf.predict(X_test_selected)))
#
#
# if __name__ == "__main__":
#     main()


################################################################################################################################################

#XGBoost

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBClassifier


def load_data(file_path):
    df = pd.read_csv(file_path)
    features = df.drop(columns=["Sample", "Condition"])

    # --- Quantile normalization ---
    features_qn = pd.DataFrame(
        quantile_transform(features, axis=0, n_quantiles=100,
                           output_distribution='normal', copy=True),
        index=features.index, columns=features.columns
    )

    # --- Z-score normalization ---
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features_qn),
        index=features_qn.index, columns=features_qn.columns
    )

    y = df["Condition"].map({"ASD": 1, "Control": 0})
    return features_scaled, y, df


def main():
    # Step 1: Load dataset (normalized + scaled)
    file_path = "ML_ready_dataset_filtered.csv"
    X, y, df = load_data(file_path)

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Step 3: Handle imbalance with SMOTE
    smote = SMOTE(sampling_strategy="auto", k_neighbors=8, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", y_train.value_counts())
    print("After SMOTE:", y_train_res.value_counts())

    # Step 4: Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=50)
    X_train_selected = selector.fit_transform(X_train_res, y_train_res)
    X_test_selected = selector.transform(X_test)

    selected_features = X.columns[selector.get_support()]
    print(f"Number of features after selection: {X_train_selected.shape[1]}")

    # Step 5: Hyperparameter search space for XGBoost
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),  # ~0.05–1.0
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'gamma': hp.uniform('gamma', 0, 5),
    }

    def objective(params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])

        clf = XGBClassifier(
            **params,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )

        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in folds.split(X_train_selected, y_train_res):
            x_tr, x_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
            y_tr, y_val_fold = y_train_res[train_idx], y_train_res[val_idx]

            clf.fit(x_tr, y_tr)
            y_pred = clf.predict(x_val_fold)
            f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))

        return {'loss': -np.mean(f1_scores), 'status': STATUS_OK, 'params': params}

    # Step 6: Run Hyperopt
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=30, trials=trials, rstate=np.random.default_rng(42))
    print("\nBest Hyperparameters:", best)

    best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
    best_params = best_trial['params']

    # Step 7: Train final XGBoost model
    clf = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric="logloss")
    clf.fit(X_train_selected, y_train_res)

    # Step 8: Evaluate on test set
    print("\nBest XGBoost Params:", best_params)
    print("Train Accuracy:", accuracy_score(y_train_res, clf.predict(X_train_selected)))
    print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test_selected)))
    print("\nClassification Report (Test):")
    print(classification_report(y_test, clf.predict(X_test_selected), target_names=["Control", "ASD"]))
    print(confusion_matrix(y_test, clf.predict(X_test_selected)))


if __name__ == "__main__":
    main()
