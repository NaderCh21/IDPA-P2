import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from XML_Similarity import *  # This should contain the necessary functions for similarity computation


# Function to convert a text file to a weighted XML file
def convert_text_to_weighted_xml(input_file_path, output_file_path):
    def parse_document(file_path):
        with open(file_path, "r") as file:
            return file.readlines()

    def create_xml_structure(lines):
        root = ET.Element("Document")
        current_section = None
        current_subsection = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                section = ET.SubElement(root, "Section", weight="1.0")
                section_title = ET.SubElement(section, "Title")
                section_title.text = line
                current_section = section
                current_subsection = None
            elif "1.1." in line or "2.1." in line:
                subsection = ET.SubElement(current_section, "Subsection", weight="0.8")
                subsection_title = ET.SubElement(subsection, "Title")
                subsection_title.text = line
                current_subsection = subsection
            else:
                parent_element = (
                    current_subsection if current_subsection else current_section
                )
                if not parent_element:
                    parent_element = root
                content = ET.SubElement(parent_element, "Content", weight="0.5")
                content.text = line

        return root

    lines = parse_document(input_file_path)
    xml_root = create_xml_structure(lines)

    tree = ET.ElementTree(xml_root)
    tree.write(output_file_path, encoding="utf-8", xml_declaration=True)


# Function to parse XML and extract weights
def parse_xml_and_extract_weights(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    weights = {}
    for element in root.iter():
        if element.tag in ["Title", "Section", "Subsection", "Content"]:
            text = element.text.strip() if element.text else ""
            weight = float(element.attrib.get("weight", 1))  # Default weight is 1
            weights[text] = weight
    return weights


# Function to compute XML similarity matrix
def compute_xml_similarity_matrix(xml_paths, vectorize_and_compute_similarity):
    n = len(xml_paths)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        weights_i = parse_xml_and_extract_weights(xml_paths[i])
        for j in range(i + 1, n):
            weights_j = parse_xml_and_extract_weights(xml_paths[j])
            similarity, _ = vectorize_and_compute_similarity(
                xml_paths[i], xml_paths[j], weights_i, weights_j
            )
            similarity_matrix[i, j] = similarity_matrix[j, i] = 1 - similarity

    return similarity_matrix


# Function to perform k-means clustering on similarity matrix
def kmeans_clustering_on_similarity_matrix(
    similarity_matrix, num_clusters, max_iterations=100
):
    kmeans = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations,
        random_state=42,
        algorithm="full",
    )
    clusters = kmeans.fit_predict(similarity_matrix)
    return clusters


# Function to plot cluster results
def plot_cluster_results(xml_paths, cluster_assignments):
    unique_clusters = np.unique(cluster_assignments)
    colors = plt.cm.tab10.colors
    symbols = ["o", "s", "^", "D", "v", ">", "<", "p", "*", "+"]

    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        cluster_label = f"Cluster {cluster_id + 1}"
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


# Main execution
def main():
    text_file_paths = [
        "python/text_files/example1.txt",
        "python/text_files/document_search_tool_1.txt",
        "python/text_files/document_search_tool_2.txt",
        "python/text_files/document_search_tool_3.txt",
    ]

    # Convert text files to XML
    xml_file_paths = []
    for text_file in text_file_paths:
        xml_file = text_file.replace(".txt", ".xml")
        convert_text_to_weighted_xml(text_file, xml_file)
        xml_file_paths.append(xml_file)

    # Compute XML similarity matrix
    similarity_matrix = compute_xml_similarity_matrix(
        xml_file_paths, vectorize_and_compute_similarity
    )

    # Specify the number of clusters
    num_clusters = 3

    # Perform k-means clustering on the similarity matrix
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
