from semi_to_xml import convert_text_to_weighted_xml
from XML_Similarity import vectorize_and_compute_similarity
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from docx import Document


def read_text_file(file_path):
    with open(file_path, "r") as file:
        return file.readlines()


def read_docx_file(file_path):
    doc = Document(file_path)
    return [paragraph.text for paragraph in doc.paragraphs if paragraph.text]


def convert_file_to_text(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == ".txt":
        return read_text_file(file_path)
    elif file_extension.lower() == ".docx":
        return read_docx_file(file_path)
    else:
        raise ValueError("Unsupported file format: " + file_extension)


# Function to compute the similarity matrix for a list of XML files
def compute_xml_similarity_matrix(xml_paths):
    n = len(xml_paths)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            similarity = vectorize_and_compute_similarity(xml_paths[i], xml_paths[j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = 1 - similarity

    return similarity_matrix


# Function for K-Means clustering
def kmeans_clustering_on_similarity_matrix(similarity_matrix, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(similarity_matrix)
    return clusters


# Function for plotting cluster results
def plot_cluster_results(xml_paths, cluster_assignments):
    unique_clusters = np.unique(cluster_assignments)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        for idx in cluster_indices:
            plt.scatter(
                cluster_id,
                idx,
                color=colors[cluster_id],
                label=f"Cluster {cluster_id+1}" if idx == cluster_indices[0] else "",
            )

    plt.xlabel("Cluster")
    plt.ylabel("Document Index")
    plt.title("KMeans Clustering Results")
    plt.legend()
    plt.show()


# Main execution
def main():
    # List of text file paths
    text_file_paths = [
        "python/text_files/example1.txt",
        "python/text_files/example2.txt",
        "python/text_files/example3.txt",
        "python/text_files/example4.txt",
    ]

    # Convert text files to XML
    xml_file_paths = []
    for text_file in text_file_paths:
        lines = convert_file_to_text(text_file)
        xml_file = os.path.splitext(text_file)[0] + ".xml"
        convert_text_to_weighted_xml(text_file, xml_file)
        xml_file_paths.append(xml_file)

    # Compute XML similarity matrix
    similarity_matrix = compute_xml_similarity_matrix(xml_file_paths)

    # Ask the user for the number of clusters
    while True:
        try:
            num_clusters = int(input("Enter the number of clusters: "))
            if num_clusters > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Perform K-Means clustering on the similarity matrix
    cluster_assignments = kmeans_clustering_on_similarity_matrix(
        similarity_matrix, num_clusters
    )

    # Plot the clustering results
    plot_cluster_results(xml_file_paths, cluster_assignments)

    # Print the clustering results
    print("Clustering Results:")
    for i, cluster_id in enumerate(cluster_assignments):
        print(f"XML File: {xml_file_paths[i]}, Cluster: {cluster_id + 1}")


if __name__ == "__main__":
    main()
