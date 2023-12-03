import numpy as np
from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET
from XML_Similarity import *  # Assuming this contains the required functions like normalize_text, get_document_fields, vectorize_and_compute_similarity
import matplotlib.pyplot as plt


def compute_xml_similarity_matrix(xml_paths, vectorize_and_compute_similarity):
    n = len(xml_paths)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            similarity, _ = vectorize_and_compute_similarity(xml_paths[i], xml_paths[j])
            # Use 1 - similarity to represent dissimilarity (lower values mean more similarity)
            similarity_matrix[i, j] = similarity_matrix[j, i] = 1 - similarity

    return similarity_matrix


def kmeans_clustering_on_similarity_matrix(
    similarity_matrix, num_clusters, max_iterations=100
):
    # Apply k-means clustering on the similarity matrix
    kmeans = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations,
        random_state=42,
        algorithm="full",
    )
    clusters = kmeans.fit_predict(similarity_matrix)

    return clusters


def plot_cluster_results(xml_paths, cluster_assignments):
    unique_clusters = np.unique(cluster_assignments)

    # Create a color map and symbols for each cluster
    colors = plt.cm.tab10.colors
    symbols = ["o", "s", "^", "D", "v", ">", "<", "p", "*", "+"]

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        cluster_label = f"Cluster {cluster_id + 1}"

        # Plot each document in the cluster
        for idx in cluster_indices:
            plt.scatter(
                cluster_id,
                idx,
                marker=symbols[cluster_id % len(symbols)],
                color=colors[cluster_id % len(colors)],
                label=cluster_label,
            )

    plt.xlabel("Cluster")
    plt.ylabel("Document")
    plt.title("KMeans Clustering Results")
    plt.legend()
    plt.show()


xml_paths = [
    "../testingfiles/city1.xml",
    "../testingfiles/city2.xml",
    "../testingfiles/city3.xml",
    "../testingfiles/city4.xml",
    "../testingfiles/food1.xml",
    "../testingfiles/food2.xml",
]

# Compute XML similarity matrix
similarity_matrix = compute_xml_similarity_matrix(
    xml_paths, vectorize_and_compute_similarity
)

# Specify the number of clusters
num_clusters = 2

# Perform k-means clustering on the similarity matrix
cluster_assignments = kmeans_clustering_on_similarity_matrix(
    similarity_matrix, num_clusters
)

# Plot the clustering results
plot_cluster_results(xml_paths, cluster_assignments)

# Print the clustering results
print("Clustering Results:")
for i, cluster_id in enumerate(cluster_assignments):
    print(f"XML File: {xml_paths[i]}, Cluster: {cluster_id + 1}")
