import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
from test import *
import matplotlib.pyplot as plt


def collect_texts(xml_paths):
    all_texts = []

    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        field_texts = []

        for element in root.iter():
            if len(element) == 0:
                text = normalize_text(element.text.strip()) if element.text else ""
                field_texts.append(text)

        all_texts.append(" ".join(field_texts))

    return all_texts


def kmeans_clustering(xml_paths, num_clusters, max_iterations=100):
    # Collect text data from XML fields for each document
    all_texts = collect_texts(xml_paths)

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_texts)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations, random_state=42)
    clusters = kmeans.fit_predict(X)

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


# Specify the paths of the XML files
xml_file_paths = [
    "testingfiles\city1.xml",
    "testingfiles\city1.xml",
    "testingfiles\city3.xml",
    "testingfiles\city4.xml",
    "testingfiles\city5.xml",
]

# Specify the number of clusters
num_clusters = 3

# Perform k-means clustering
cluster_assignments = kmeans_clustering(xml_file_paths, num_clusters)

# Plot the clustering results
plot_cluster_results(xml_file_paths, cluster_assignments)
