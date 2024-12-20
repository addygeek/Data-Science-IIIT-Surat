import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# Sample dataset
data = {
    "Review": [
        "I love this product, it’s amazing!",
        "Terrible experience, would not recommend.",
        "Quite good, but has some minor issues.",
        "Absolutely fantastic, exceeded my expectations!",
        "Not worth the price.",
        "Very happy with this purchase!",
        "It’s okay, but I’ve seen better.",
        "The product broke after a week, disappointed."
    ]
}
df = pd.DataFrame(data)

print("Original Reviews:\n", df)

# Sentiment Analysis
def analyze_sentiment(review):
    analysis = TextBlob(review)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative"

df['Sentiment'] = df['Review'].apply(analyze_sentiment)

print("\nSentiment Analysis Results:\n", df)

# Visualize Sentiment Distribution
sns.countplot(data=df, x='Sentiment', palette='viridis')
plt.title("Sentiment Distribution")
plt.show()

# Generate Word Cloud for Positive Reviews
positive_reviews = " ".join(df[df['Sentiment'] == 'Positive']['Review'])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(positive_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Positive Reviews")
plt.show()

# Most Common Words
vectorizer = CountVectorizer(stop_words='english')
word_count = vectorizer.fit_transform(df['Review'])
word_sum = np.sum(word_count.toarray(), axis=0)

words_df = pd.DataFrame({
    "Word": vectorizer.get_feature_names_out(),
    "Frequency": word_sum
}).sort_values(by="Frequency", ascending=False)

print("\nMost Common Words:\n", words_df.head())

# Save Results
df.to_csv("sentiment_analysis_results.csv", index=False)
print("\nSentiment Analysis results saved to 'sentiment_analysis_results.csv'.")
