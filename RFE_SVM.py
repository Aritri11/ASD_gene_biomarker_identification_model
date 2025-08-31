import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """
    Load dataset and prepare features (X) and target (y).
    Returns:
        X (DataFrame): Feature matrix
        y (Series): Target labels (1=ASD, 0=Control)
        df (DataFrame): Original dataset
    """
    df = pd.read_csv(file_path)
    X = df.drop(columns=["Sample", "Condition"])
    y = df["Condition"].map({"ASD": 1, "Control": 0})
    return X, y, df


def main():
    # Step 1: Load dataset
    file_path = "C:/Users/Aritri Baidya/Desktop/ML Project/R_codes/ML_ready_dataset_filtered.csv"
    X, y, df = load_data(file_path)

    # Step 2: Define SVM model

    svm_model = SVC(kernel="linear", C=0.01)


    # Step 3: Apply RFE for feature selection

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    selector = RFE(estimator=svm_model, n_features_to_select=50, step=5)
    selector.fit(X, y)


    # Step 4: Extract selected features

    print(f"Optimal number of features: {selector.n_features_}")
    selected_features = X.columns[selector.support_]
    print("Selected Features:")
    print(selected_features.tolist())

    # Step 5: Create new dataset with selected genes

    df_selected = df[["Sample"] + selected_features.tolist() + ["Condition"]]
    df_selected.to_csv(
        "C:/Users/Aritri Baidya/Desktop/ML Project/R_codes/Selected_Genes_Expression.csv",
        index=False,
    )
    print("New dataset with selected genes saved as 'Selected_Genes_Expression.csv'")


    # Step 6: Heatmap of selected genes

    data_matrix = df_selected.set_index("Sample")
    conditions = data_matrix["Condition"]
    data_matrix = data_matrix.drop(columns="Condition")

    # Color labels for ASD vs Control
    condition_colors = conditions.map({"ASD": "red", "Control": "blue"})

    # Plot clustermap
    sns.clustermap(
        data_matrix,
        cmap="vlag",
        standard_scale=1,     # normalize each gene across samples
        row_colors=condition_colors,
        figsize=(14, 10)
    )

    plt.suptitle("Heatmap of Selected Genes (SVM-RFE)", y=1.02)
    plt.savefig("Selected_Genes_Heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Heatmap displayed (can also be saved as PNG).")


if __name__ == "__main__":
    main()
