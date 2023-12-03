import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
import numpy as np


# Function to normalize and preprocess text
def normalize_text(text):
    # Tokenize the input text into individual words
    tokens = word_tokenize(text)

    # Convert all tokens to lowercase
    tokens = [w.lower() for w in tokens]

    # Create a translation table to remove punctuation
    table = str.maketrans("", "", string.punctuation)

    # Remove punctuation from each token using the translation table
    stripped = [w.translate(table) for w in tokens]

    # Filter out tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # Define a set of English stopwords
    stop_words = set(stopwords.words("english"))

    # Remove stopwords from the list of tokens
    words = [w for w in words if not w in stop_words]

    # Initialize an empty list to store lemmatized words
    lemmatized = []

    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Iterate over each word and its part-of-speech tag
    for word, tag in pos_tag(words):
        # Check the part-of-speech tag and lemmatize accordingly
        if tag.startswith("V"):  # Verb
            word = lemmatizer.lemmatize(word, "v")
        elif tag.startswith("J"):  # Adjective
            word = lemmatizer.lemmatize(word, "a")
        elif tag.startswith("N"):  # Noun
            word = lemmatizer.lemmatize(word, "n")
        elif tag.startswith("R"):  # Adverb
            word = lemmatizer.lemmatize(word, "r")

        # Append the lemmatized word to the list
        lemmatized.append(word)

    # Return the list of lemmatized words as a space-separated string
    return " ".join(lemmatized)


def get_document_fields(xml_path):
    # Parse the XML file and get a list of unique fields
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fields = set(element.tag for element in root.iter() if len(element) == 0)

    return fields


def vectorize_and_compute_similarity(xml_path1, xml_path2):
    # Get fields for both documents
    fields1 = get_document_fields(xml_path1)
    fields2 = get_document_fields(xml_path2)

    # Display common and unique fields
    common_fields = fields1 & fields2
    unique_fields1 = fields1 - fields2
    unique_fields2 = fields2 - fields1

    print(f"Common Fields: {common_fields}")
    print(f"Unique Fields in Document 1: {unique_fields1}")
    print(f"Unique Fields in Document 2: {unique_fields2}")

    # User input: Specify weights for each field
    field_weights = {}
    for field in common_fields:
        weight = float(input(f"Enter the weight for field '{field}': "))
        field_weights[field] = weight

    # Parse the first XML file
    tree1 = ET.parse(xml_path1)
    root1 = tree1.getroot()

    # Parse the second XML file
    tree2 = ET.parse(xml_path2)
    root2 = tree2.getroot()

    # Collect all text data from XML fields for both files
    field_texts1 = {}
    field_texts2 = {}

    for element in root1.iter():
        if len(element) == 0:
            field_name = element.tag
            field_texts1[field_name] = field_texts1.get(field_name, []) + [
                normalize_text(element.text.strip()) if element.text else ""
            ]

    for element in root2.iter():
        if len(element) == 0:
            field_name = element.tag
            field_texts2[field_name] = field_texts2.get(field_name, []) + [
                normalize_text(element.text.strip()) if element.text else ""
            ]

    # Combine texts from both documents for each field
    combined_texts = {}
    for field_name in common_fields:
        combined_texts[field_name] = field_texts1.get(
            field_name, []
        ) + field_texts2.get(field_name, [])

    # Print the terms and field vectors for each document
    print("\nDocument 1:")
    for field_name, texts in combined_texts.items():
        print(f"\n{field_name} Terms: {texts}")
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        print(f"{field_name} Vectors: {X.toarray()}")

    print("\nDocument 2:")
    for field_name, texts in combined_texts.items():
        print(f"\n{field_name} Terms: {texts}")
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        print(f"{field_name} Vectors: {X.toarray()}")

    # Vectorize and compute cosine similarity for each field
    field_similarities = {}
    final_similarity_score = 0.0

    for field_name, texts in combined_texts.items():
        # Vectorize the preprocessed text data for each field
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)
        vectors = X.toarray()

        # Separate vectors for each document
        vectors1 = vectors[: len(field_texts1.get(field_name, []))]
        vectors2 = vectors[len(field_texts1.get(field_name, [])) :]

        # Compute cosine similarity for each pair of vectors
        similarities = cosine_similarity(vectors1, vectors2)

        # Compute the final similarity score for the field
        field_similarity = field_weights.get(field_name, 1.0) * np.mean(similarities)
        final_similarity_score += field_similarity

        # Store the cosine similarity for the field
        field_similarities[field_name] = similarities.tolist()

    return final_similarity_score, field_similarities


# # Specify the paths of the two XML files
# xml_file_path1 = "testingfiles\city2.xml"
# xml_file_path2 = "testingfiles\city5.xml"

# # Vectorize and compute cosine similarity
# final_similarity, field_similarities = vectorize_and_compute_similarity(
#     xml_file_path1, xml_file_path2
# )

# # Print the terms and vectors for each field
# for field_name, similarity_matrix in field_similarities.items():
#     print(f"\n{field_name} Cosine Similarity:")
#     print(similarity_matrix)

# # Print the final similarity score
# print(f"\nFinal Similarity Score: {final_similarity}")
