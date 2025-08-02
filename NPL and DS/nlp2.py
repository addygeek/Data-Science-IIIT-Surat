import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nlp = spacy.load("en_core_web_sm")

# Sample documents
documents = [
    "Natural language processing enables computers to understand human language.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning algorithms are inspired by the human brain.",
    "NLP tasks include text classification and sentiment analysis.",
    "Clustering is an unsupervised machine learning technique."
]


def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

processed_docs = [preprocess(doc) for doc in documents]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_docs)

# KMeans Clustering
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Output clusters
for i, doc in enumerate(documents):
    print(f"Document: {doc}")
    print(f"Cluster: {kmeans.labels_[i]}")
    print("-" * 40)