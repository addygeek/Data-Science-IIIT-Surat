import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

def summarize_text(text, num_sentences=3):
    """
    Summarize a given text by extracting the most important sentences using TF-IDF.
    :param text: Input text to summarize
    :param num_sentences: Number of sentences for the summary
    :return: Summarized text
    """
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Rank sentences based on scores
    ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[-num_sentences:]]
    
    # Combine the top sentences into a summary
    summary = " ".join(ranked_sentences)
    return summary

# Example Usage
if __name__ == "__main__":
    input_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create machines that mimic human intelligence.
    AI has been widely adopted in numerous industries, including healthcare, finance, and transportation.
    In healthcare, AI helps diagnose diseases and create personalized treatment plans.
    In finance, AI algorithms analyze market trends to automate trading strategies.
    Autonomous vehicles are one of the key applications of AI in transportation.
    As AI continues to evolve, it raises ethical questions about its impact on jobs and society.
    Experts emphasize the need for regulations to ensure responsible use of AI technologies.
    """
    
    print("Original Text:")
    print(input_text)
    print("\nSummarized Text:")
    print(summarize_text(input_text, num_sentences=3))
