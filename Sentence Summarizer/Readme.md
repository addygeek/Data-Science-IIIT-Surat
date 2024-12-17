# Text Summarizer Using TF-IDF

This project is a Python-based text summarization tool that extracts the most important sentences from input text using the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm.

## Features

1. **Text Tokenization**:
   - Splits the input text into sentences using the NLTK library.

2. **TF-IDF Scoring**:
   - Computes TF-IDF scores to rank sentences based on their importance.

3. **Summarization**:
   - Extracts the top N sentences that represent the most crucial information in the text.

4. **Customizable Summary Length**:
   - Allows users to specify how many sentences should be included in the summary.

## Dependencies

- `nltk`: Natural Language Toolkit for text tokenization and stopwords.
- `scikit-learn`: For TF-IDF vectorization.
- `numpy`: For matrix manipulation.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
