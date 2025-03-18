import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def extractive_summarization(text, num_sentences=3):
    """
    Generate an extractive summary by selecting the most important sentences from the text.

    Args:
        text (str): The input text to summarize.
        num_sentences (int): Number of sentences to include in the summary.

    Returns:
        str: The generated summary.
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Check if the text is too short to summarize
    if len(sentences) <= num_sentences:
        return text

    # Preprocess sentences
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)

    # Compute similarity matrix
    similarity_matrix = np.dot(sentence_vectors, sentence_vectors.T).toarray()

    # Build a graph and rank sentences using PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # Rank sentences by their scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Select the top-ranked sentences
    summary_sentences = [ranked_sentences[i][1] for i in range(num_sentences)]
    summary = ' '.join(summary_sentences)

    return summary

# Example usage
if __name__ == "__main__":
    # Specify the path to the input file in another folder
    input_file = "data/sample text.txt"  # Adjust the path as needed
    with open(input_file, "r", encoding="utf-8") as file:
        input_text = file.read()

    # Generate summary
    summary = extractive_summarization(input_text, num_sentences=5)
    print("Extractive Summary:")
    print(summary)