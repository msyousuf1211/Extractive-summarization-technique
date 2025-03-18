import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

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

    sentences = sent_tokenize(text)


    if len(sentences) <= num_sentences:
        return text
    
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)

    
    similarity_matrix = np.dot(sentence_vectors, sentence_vectors.T).toarray()

  
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

  
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summary_sentences = [ranked_sentences[i][1] for i in range(num_sentences)]
    summary = ' '.join(summary_sentences)

    return summary

if __name__ == "__main__":
    
    input_file = "data/sample text.txt"  
    with open(input_file, "r", encoding="utf-8") as file:
        input_text = file.read()

   
    summary = extractive_summarization(input_text, num_sentences=5)
    print("Extractive Summary:")
    print(summary)