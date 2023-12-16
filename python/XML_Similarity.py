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


def get_document_fields_and_weights(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fields = {}
    for element in root.iter():
        if len(element) == 0:
            weight = float(element.attrib.get("weight", 1))
            fields[element.tag] = weight

    return fields


def vectorize_and_compute_similarity(xml_path1, xml_path2):
    fields1 = get_document_fields_and_weights(xml_path1)
    fields2 = get_document_fields_and_weights(xml_path2)

    common_fields = set(fields1.keys()) & set(fields2.keys())

    tree1 = ET.parse(xml_path1)
    root1 = tree1.getroot()

    tree2 = ET.parse(xml_path2)
    root2 = tree2.getroot()

    field_texts1 = {}
    field_texts2 = {}

    for element in root1.iter():
        if element.tag in common_fields:
            field_texts1[element.tag] = (
                field_texts1.get(element.tag, "")
                + " "
                + normalize_text(element.text.strip() if element.text else "")
            )

    for element in root2.iter():
        if element.tag in common_fields:
            field_texts2[element.tag] = (
                field_texts2.get(element.tag, "")
                + " "
                + normalize_text(element.text.strip() if element.text else "")
            )

    final_similarity_score = 0.0

    for field in common_fields:
        vectorizer = CountVectorizer()
        combined_text = [field_texts1[field], field_texts2[field]]
        vectors = vectorizer.fit_transform(combined_text).toarray()

        similarity = cosine_similarity([vectors[0]], [vectors[1]])[0, 0]

        weighted_similarity = similarity * min(fields1[field], fields2[field])
        final_similarity_score += weighted_similarity

    return final_similarity_score


# # Example usage
# xml_file_path1 = "python/text_files/document_search_tool_1.xml"
# xml_file_path2 = "python/text_files/document_search_tool_2.xml"
# similarity_score = vectorize_and_compute_similarity(xml_file_path1, xml_file_path2)
# print(f"Similarity Score: {similarity_score}")
