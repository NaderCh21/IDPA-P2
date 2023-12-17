import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from XML_Similarity import *

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


def hierarchical_clustering_on_similarity_matrix(similarity_matrix, num_clusters):
    # Use hierarchical agglomerative clustering
    linkage_matrix = linkage(similarity_matrix, method='ward')
    clustering_model = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    clusters = clustering_model.fit_predict(linkage_matrix)

    return clusters

def plot_dendrogram(similarity_matrix, xml_paths):
    linkage_matrix = linkage(similarity_matrix, method='ward')
    dendrogram(linkage_matrix, labels=[xml_path.split('/')[-1] for xml_path in xml_paths])
    plt.title("Hierarchical Agglomerative Clustering Dendrogram")
    plt.xlabel("Document")
    plt.ylabel("Distance")
    plt.show()

if __name__ == "__main__":
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
    num_clusters_hierarchical = 2

    # Perform hierarchical agglomerative clustering on the similarity matrix
    cluster_assignments_hierarchical = hierarchical_clustering_on_similarity_matrix(
        similarity_matrix, num_clusters_hierarchical
    )

 

    # Print the hierarchical agglomerative clustering results
    print("Hierarchical Agglomerative Clustering Results:")
    for i, cluster_id in enumerate(cluster_assignments_hierarchical):
        print(f"XML File: {xml_paths[i]}, Cluster: {cluster_id + 1}")

    # Plot the hierarchical agglomerative clustering dendrogram
    plot_dendrogram(similarity_matrix, xml_paths)
