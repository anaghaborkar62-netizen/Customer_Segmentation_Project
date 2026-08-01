import streamlit as st
import pandas as pd
import sqlite3
import json
import time

from datetime import datetime

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from clustering_pipeline import (
    load_dataset,
    get_numeric_columns,
    preprocess,
    find_best_k,
    plot_elbow_and_silhouette,
    run_kmeans,
    plot_clusters,
    get_cluster_ranges
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")

st.write(
"""
Perform customer segmentation using
**K-Means Clustering** and monitor every execution.
"""
)

# =====================================================
# DATABASE
# =====================================================

conn = sqlite3.connect(
    "customer_segmentation_v2.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS clustering_history(

id INTEGER PRIMARY KEY AUTOINCREMENT,

created_at TEXT,

dataset_name TEXT,

selected_features TEXT,

number_of_clusters INTEGER,

silhouette_score REAL,

davies_score REAL,

calinski_score REAL,

execution_time REAL,

cluster_summary TEXT

)
""")

conn.commit()

# =====================================================
# SAVE HISTORY
# =====================================================

def save_history(
    dataset_name,
    features,
    k,
    silhouette,
    davies,
    calinski,
    execution_time,
    labels
):

    summary = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    cursor.execute(
        """
        INSERT INTO clustering_history
        (
        created_at,
        dataset_name,
        selected_features,
        number_of_clusters,
        silhouette_score,
        davies_score,
        calinski_score,
        execution_time,
        cluster_summary
        )

        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_name,
            ", ".join(features),
            k,
            float(silhouette),
            float(davies),
            float(calinski),
            float(execution_time),
            json.dumps(summary)
        )
    )

    conn.commit()

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file.")
    st.stop()

# =====================================================
# LOAD DATASET
# =====================================================

df = load_dataset(uploaded_file)

st.subheader("Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

col1, col2 = st.columns(2)

col1.metric(
    "Rows",
    df.shape[0]
)

col2.metric(
    "Columns",
    df.shape[1]
)

numeric_columns = get_numeric_columns(df)

if len(numeric_columns) < 2:

    st.error(
        "Dataset must contain at least two numeric columns."
    )

    st.stop()
# =====================================================
# FEATURE SELECTION
# =====================================================

st.markdown("---")
st.header("⚙ Feature Selection")

selected_features = st.multiselect(
    "Select Numeric Features",
    numeric_columns,
    default=numeric_columns[:2]
)

if len(selected_features) < 2:
    st.warning("Please select at least two numeric features.")
    st.stop()

# =====================================================
# PREPROCESS DATA
# =====================================================

X_scaled, scaler, clean_df = preprocess(
    df,
    selected_features
)

st.success(
    f"Selected Features: {', '.join(selected_features)}"
)

# =====================================================
# FIND BEST K
# =====================================================

st.markdown("---")
st.header("📈 Find Optimal Number of Clusters")

max_k = min(10, len(clean_df) - 1)

best_k, inertia, silhouette_scores, k_values = find_best_k(
    X_scaled,
    max_k=max_k
)

fig = plot_elbow_and_silhouette(
    k_values,
    inertia,
    silhouette_scores
)

st.pyplot(fig)

st.success(f"Suggested Best K = {best_k}")

# =====================================================
# USER SELECTS K
# =====================================================

k = st.slider(
    "Choose Number of Clusters",
    min_value=2,
    max_value=max_k,
    value=best_k
)

# =====================================================
# RUN CLUSTERING
# =====================================================

run = st.button(
    "🚀 Run Clustering",
    use_container_width=True
)

if run:

    start_time = time.time()

    labels, model, silhouette = run_kmeans(
        X_scaled,
        k
    )

    execution_time = round(
        time.time() - start_time,
        3
    )

    davies = davies_bouldin_score(
        X_scaled,
        labels
    )

    calinski = calinski_harabasz_score(
        X_scaled,
        labels
    )

    save_history(
        uploaded_file.name,
        selected_features,
        k,
        silhouette,
        davies,
        calinski,
        execution_time,
        labels
    )

    st.success("✅ Clustering Completed Successfully")

    # =====================================================
    # METRICS
    # =====================================================

    st.markdown("---")
    st.header("📊 Clustering Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Silhouette Score",
        round(silhouette, 4)
    )

    col2.metric(
        "Davies-Bouldin",
        round(davies, 4)
    )

    col3.metric(
        "Calinski-Harabasz",
        round(calinski, 2)
    )

    col4.metric(
        "Execution Time (sec)",
        execution_time
    )
    # =====================================================
    # CLUSTER VISUALIZATION
    # =====================================================

    st.markdown("---")
    st.header("📈 Customer Cluster Visualization")

    fig = plot_clusters(
        X_scaled,
        labels,
        selected_features
    )

    st.pyplot(fig)

    # =====================================================
    # CLUSTER SUMMARY
    # =====================================================

    st.markdown("---")
    st.header("📋 Cluster Summary")

    summary = get_cluster_ranges(
        clean_df,
        selected_features,
        labels
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    # =====================================================
    # CUSTOMER COUNT PER CLUSTER
    # =====================================================

    st.markdown("---")
    st.header("👥 Customers in Each Cluster")

    cluster_counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    cluster_df = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Customers": cluster_counts.values
    })

    st.dataframe(
        cluster_df,
        use_container_width=True
    )

    st.bar_chart(
        cluster_df.set_index("Cluster")
    )

    # =====================================================
    # CLUSTERED DATASET
    # =====================================================

    result_df = clean_df.copy()

    result_df["Cluster"] = labels

    st.markdown("---")
    st.header("📄 Clustered Dataset")

    st.dataframe(
        result_df,
        use_container_width=True,
        height=400
    )

    # =====================================================
    # DOWNLOAD CLUSTERED DATA
    # =====================================================

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Clustered Dataset",
        data=csv,
        file_name="clustered_dataset.csv",
        mime="text/csv",
        key="download_clustered_dataset"
    )
# =====================================================
# PREVIOUS CLUSTERING HISTORY
# =====================================================

st.markdown("---")
st.header("📜 Previous Clustering History")

history = pd.read_sql_query(
    """
    SELECT *
    FROM clustering_history
    ORDER BY id DESC
    """,
    conn
)

if history.empty:

    st.info("No previous clustering runs found.")

else:

    history_display = []

    for _, row in history.iterrows():

        try:
            summary = json.loads(row["cluster_summary"])
        except:
            summary = {}

        history_row = {
            "Time": row["created_at"],
            "Dataset": row["dataset_name"],
            "Features": row["selected_features"],
            "Clusters": row["number_of_clusters"],
            "Silhouette": round(row["silhouette_score"], 4),
            "Davies": round(row["davies_score"], 4),
            "Calinski": round(row["calinski_score"], 2),
            "Execution Time (sec)": round(row["execution_time"], 3)
        }

        if isinstance(summary, dict):
            for cluster, count in summary.items():
                history_row[f"Cluster {cluster}"] = count

        history_display.append(history_row)

    history_df = pd.DataFrame(history_display)

    st.dataframe(
        history_df,
        use_container_width=True,
        height=350
    )

    st.download_button(
        label="📥 Download History",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="clustering_history.csv",
        mime="text/csv",
        key="download_history"
    )

    if st.button(
        "🗑 Clear History",
        key="clear_history"
    ):

        cursor.execute(
            "DELETE FROM clustering_history"
        )

        conn.commit()

        st.success("History Cleared Successfully.")

        st.rerun()
# =====================================================
# MONITORING DASHBOARD
# =====================================================

st.markdown("---")
st.header("📊 Monitoring Dashboard")

history = pd.read_sql_query(
    """
    SELECT *
    FROM clustering_history
    ORDER BY id DESC
    """,
    conn
)

if history.empty:

    st.info("No monitoring data available.")

else:

    total_runs = len(history)

    best_silhouette = history["silhouette_score"].max()

    avg_silhouette = round(
        history["silhouette_score"].mean(),
        4
    )

    avg_davies = round(
        history["davies_score"].mean(),
        4
    )

    avg_calinski = round(
        history["calinski_score"].mean(),
        2
    )

    avg_time = round(
        history["execution_time"].mean(),
        3
    )

    last_run = history.iloc[0]["created_at"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "📂 Total Runs",
        total_runs
    )

    col2.metric(
        "⭐ Best Silhouette",
        round(best_silhouette, 4)
    )

    col3.metric(
        "⏱ Avg Time (sec)",
        avg_time
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "📉 Avg Davies",
        avg_davies
    )

    col5.metric(
        "📈 Avg Calinski",
        avg_calinski
    )

    col6.metric(
        "🕒 Last Run",
        last_run
    )

    st.markdown("---")

    st.subheader("📈 Average Metrics")

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Silhouette",
                "Davies-Bouldin",
                "Calinski-Harabasz"
            ],
            "Value": [
                avg_silhouette,
                avg_davies,
                avg_calinski
            ]
        }
    )

    st.bar_chart(
        metrics_df.set_index("Metric")
    )

    st.markdown("---")

    st.success(
        "Monitoring dashboard updated successfully."
    )
# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📊 Customer Segmentation")

st.sidebar.info(
    """
This application performs customer segmentation
using the K-Means clustering algorithm.

Features:

• Upload CSV Dataset

• Feature Selection

• Elbow Method

• Silhouette Analysis

• Cluster Visualization

• Monitoring Dashboard

• Previous History

• SQLite Database
"""
)

st.sidebar.markdown("---")

st.sidebar.success("Ready for Clustering")

# =====================================================
# PROJECT INFORMATION
# =====================================================

st.markdown("---")

with st.expander("ℹ About this Project"):

    st.markdown("""

## Customer Segmentation using Machine Learning

This project performs customer segmentation
using the K-Means Clustering Algorithm.

### Features

- Upload CSV Dataset

- Automatic Data Preprocessing

- Feature Selection

- Elbow Method

- Best K Suggestion

- Cluster Visualization

- Cluster Summary

- Download Clustered Dataset

- SQLite Database Storage

- Previous Clustering History

- Monitoring Dashboard

### Metrics Stored

- Dataset Name

- Selected Features

- Number of Clusters

- Silhouette Score

- Davies-Bouldin Score

- Calinski-Harabasz Score

- Execution Time

- Customer Count per Cluster

- Timestamp

""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center'>

<h3>Customer Segmentation Monitoring System</h3>

Developed using

<b>Python | Streamlit | SQLite | Scikit-Learn</b>

</div>
""",
unsafe_allow_html=True
)

# =====================================================
# CLOSE DATABASE
# =====================================================

conn.close()