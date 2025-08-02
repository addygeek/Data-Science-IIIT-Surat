import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

text = """
I am aditya , and i am learning Natural Language Processing (NLP).
NLP is a fascinating field that combines computer science, artificial intelligence, and linguistics to enable
computers to understand, interpret, and generate human language.
In this example, we will explore some basic NLP tasks such as tokenization, stopword removal
, stemming, lemmatization, part-of-speech tagging, and named entity recognition.
NLP has applications in various domains including chatbots, sentiment analysis, and language translation.   
"""

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Word Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Remove Stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("Filtered Tokens:", filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_tokens]
print("Stemmed Tokens:", stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("Lemmatized Tokens:", lemmatized)

# Part-of-Speech Tagging
pos_tags = pos_tag(filtered_tokens)
print("POS Tags:", pos_tags)

# Named Entity Recognition
named_entities = ne_chunk(pos_tags)
print("Named Entities:")
print(named_entities)