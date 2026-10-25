
# Customer Segmentation Using Clustering Algorithms
# Dataset: Mall Customers from Kaggle

import kagglehub
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Step 1: Download dataset from Kaggle
path = kagglehub.dataset_download(
    "vjchoudhary7/customer-segmentation-tutorial-in-python"
)

print("Path to dataset files:", path)

# Step 2: Find and load the CSV file
csv_files = glob.glob(os.path.join(path, "*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded dataset.")

df = pd.read_csv(csv_files[0])

print("\nFirst five records:")
print(df.head())

print("\nDataset shape:", df.shape)

# Step 3: Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X = X.fillna(X.median())

# Step 4: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Define clustering algorithms
models = {
    "K-Means": KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    ),
    "Agglomerative Clustering": AgglomerativeClustering(
        n_clusters=5
    ),
    "DBSCAN": DBSCAN(
        eps=0.5,
        min_samples=5
    )
}

results = []
cluster_labels = {}

# Step 6: Apply clustering and evaluate
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    cluster_labels[name] = labels

    # Exclude DBSCAN noise points from evaluation
    mask = labels != -1
    valid_labels = labels[mask]
    valid_data = X_scaled[mask]

    n_clusters = len(set(valid_labels))

    if 2 <= n_clusters < len(valid_labels):
        silhouette = silhouette_score(
            valid_data, valid_labels
        )
        db_index = davies_bouldin_score(
            valid_data, valid_labels
        )
    else:
        silhouette = None
        db_index = None

    results.append({
        "Algorithm": name,
        "Number of Clusters": n_clusters,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index
    })

    print(f"\n{name}")
    print("Number of clusters:", n_clusters)
    print("Silhouette Score:", silhouette)
    print("Davies-Bouldin Index:", db_index)

# Step 7: Display comparison table
results_df = pd.DataFrame(results)

print("\nClustering Performance Comparison:")
print(results_df.to_string(index=False))

# Step 8: Visualize all clustering algorithms
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, labels) in zip(
    axes, cluster_labels.items()
):
    ax.scatter(
        X["Annual Income (k$)"],
        X["Spending Score (1-100)"],
        c=labels,
        cmap="viridis",
        s=50,
        edgecolors="black",
        linewidths=0.3
    )

    ax.set_title(name)
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.grid(True, alpha=0.3)

plt.suptitle("Customer Segmentation Results")
plt.tight_layout()
plt.show()

# Step 9: Add K-Means cluster labels to dataset
df["KMeans_Cluster"] = cluster_labels["K-Means"]

print("\nCustomer Cluster Assignments:")
print(df.head(10))

# Step 10: Save output to CSV
output_file = "Customer_Clustering_Results.csv"
df.to_csv(output_file, index=False)

print("\nResults saved to:", output_file)
