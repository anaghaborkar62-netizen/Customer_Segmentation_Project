import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(uploaded_file):

    df = pd.read_csv(uploaded_file)

    return df


# ==========================================================
# GET NUMERIC COLUMNS
# ==========================================================

def get_numeric_columns(df):

    return df.select_dtypes(include=["number"]).columns.tolist()


# ==========================================================
# PREPROCESS DATA
# ==========================================================

def preprocess(df, selected_features):

    clean_df = df[selected_features].dropna().copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(clean_df)

    return X_scaled, scaler, clean_df


# ==========================================================
# FIND BEST K
# ==========================================================

def find_best_k(X_scaled, max_k=10):

    inertia = []

    silhouette_scores = []

    k_values = list(range(2, max_k + 1))

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        inertia.append(model.inertia_)

        silhouette_scores.append(
            silhouette_score(
                X_scaled,
                labels
            )
        )

    best_index = silhouette_scores.index(
        max(silhouette_scores)
    )

    best_k = k_values[best_index]

    return (
        best_k,
        inertia,
        silhouette_scores,
        k_values
    )
# ==========================================================
# ELBOW METHOD + SILHOUETTE PLOT
# ==========================================================

def plot_elbow_and_silhouette(
    k_values,
    inertia,
    silhouette_scores
):

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow Method
    ax[0].plot(
        k_values,
        inertia,
        marker="o",
        linewidth=2
    )

    ax[0].set_title("Elbow Method")
    ax[0].set_xlabel("Number of Clusters (K)")
    ax[0].set_ylabel("Inertia")
    ax[0].grid(True)

    # Silhouette Scores
    ax[1].plot(
        k_values,
        silhouette_scores,
        marker="o",
        linewidth=2
    )

    ax[1].set_title("Silhouette Scores")
    ax[1].set_xlabel("Number of Clusters (K)")
    ax[1].set_ylabel("Silhouette Score")
    ax[1].grid(True)

    plt.tight_layout()

    return fig


# ==========================================================
# RUN KMEANS
# ==========================================================

def run_kmeans(
    X_scaled,
    k
):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    return labels, model, silhouette


# ==========================================================
# PCA CLUSTER VISUALIZATION
# ==========================================================

def plot_clusters(
    X_scaled,
    labels,
    selected_features
):

    pca = PCA(
        n_components=2,
        random_state=42
    )

    components = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        components[:, 0],
        components[:, 1],
        c=labels
    )

    ax.set_title("Customer Segments (PCA)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    plt.colorbar(
        scatter,
        ax=ax,
        label="Cluster"
    )

    plt.tight_layout()

    return fig
# ==========================================================
# CLUSTER SUMMARY
# ==========================================================

def get_cluster_ranges(
    clean_df,
    selected_features,
    labels
):

    df = clean_df.copy()

    df["Cluster"] = labels

    summary = (
        df.groupby("Cluster")
        .agg(["mean", "min", "max"])
        .round(2)
    )

    # Flatten MultiIndex column names
    summary.columns = [
        f"{col[0]}_{col[1]}"
        for col in summary.columns
    ]

    # Customer count per cluster
    summary["Customers"] = (
        df.groupby("Cluster")
        .size()
    )

    summary.reset_index(inplace=True)

    return summary


# ==========================================================
# CLUSTER CENTERS
# ==========================================================

def get_cluster_centers(
    model,
    scaler,
    selected_features
):

    centers = scaler.inverse_transform(
        model.cluster_centers_
    )

    centers_df = pd.DataFrame(
        centers,
        columns=selected_features
    )

    centers_df.index.name = "Cluster"

    return centers_df


# ==========================================================
# CLUSTER DISTRIBUTION
# ==========================================================

def get_cluster_distribution(labels):

    distribution = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .reset_index()
    )

    distribution.columns = [
        "Cluster",
        "Customers"
    ]

    return distribution