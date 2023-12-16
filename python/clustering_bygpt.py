import numpy as np
from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET
from XML_Similarity import *  # Assuming this contains the required functions like normalize_text, get_document_fields, vectorize_and_compute_similarity
import matplotlib.pyplot as plt


def calculate_element_weights(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def assign_weights(element):
        # Assuming a weight of 1 for parent elements and 0.5 for leaf elements
        if len(element):  # If the element has children
            weight = 1
        else:
            weight = 0.5
        weights[element.tag] = weights.get(element.tag, 0) + weight
        for child in element:
            assign_weights(child)

    weights = {}
    assign_weights(root)
    return weights


def compute_xml_similarity_matrix(xml_paths, vectorize_and_compute_similarity):
    n = len(xml_paths)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        weights_i = calculate_element_weights(xml_paths[i])
        for j in range(i + 1, n):
            weights_j = calculate_element_weights(xml_paths[j])
            # Modify this function to take weights into account
            similarity, _ = vectorize_and_compute_similarity(
                xml_paths[i], xml_paths[j], weights_i, weights_j
            )
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
    "testingfiles/city1.xml",
    "testingfiles/city2.xml",
    "testingfiles/city3.xml",
    "testingfiles/city4.xml",
    "testingfiles/city5.xml",
    "testingfiles/food1.xml",
    "testingfiles/food2.xml",
]

# Ask the user for the number of clusters
while True:
    user_input = input("Enter the number of clusters: ")

    if not user_input.strip():  # Check if the input is not just whitespace or empty
        print("Input cannot be empty. Please enter a number.")
        continue

    try:
        num_clusters = int(user_input)
        if num_clusters > 0:
            break
        else:
            print("Please enter a positive integer.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

# Compute XML similarity matrix
similarity_matrix = compute_xml_similarity_matrix(
    xml_paths, vectorize_and_compute_similarity
)

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
