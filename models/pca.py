import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

vocab_cut_path = '../word_embeddings/processed_vocab_cut.txt'
embedding_path = '../word_embeddings/processed_embeddings.npy'
train_pos_path = '../twitter-datasets/train_pos_full.txt'
train_neg_path = '../twitter-datasets/train_neg_full.txt'

### DATA LOADING

# Load the word embeddings
word_embeddings = np.load(embedding_path)

# Load the vocabulary
with open(vocab_cut_path, 'r', encoding='utf-8') as file:
    vocabulary = file.read().splitlines()

# Create a dictionary to map words to their corresponding embeddings
word_to_embedding = {word: word_embeddings[i] for i, word in enumerate(vocabulary)}

# Load positive training tweets and assign labels
with open(train_pos_path, 'r', encoding='utf-8') as file:
    pos_tweets = file.readlines()

pos_labels = np.ones(len(pos_tweets), dtype=int)  # Assign label 1 for positive tweets

# Load negative training tweets and assign labels
with open(train_neg_path, 'r', encoding='utf-8') as file:
    neg_tweets = file.readlines()

neg_labels = -1 * np.ones(len(neg_tweets), dtype=int)  # Assign label -1 for negative tweets

# Combine positive and negative tweets and labels
train_tweets = pos_tweets + neg_tweets
labels = np.concatenate((pos_labels, neg_labels), axis=0)

### DEFINE FUNCTIONS

def average_word_vectors(tweet, word_to_embedding):
    words = tweet.split()
    vectors = [word_to_embedding[word] for word in words if word in word_to_embedding]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # If none of the words in the tweet are in the embeddings, return a zero vector
        return np.zeros_like(word_embeddings[0])
    
# Construct feature representations for training and testing tweets
train_features = [average_word_vectors(tweet, word_to_embedding) for tweet in train_tweets]

# Plot PCA result with labels, using different markers for each class
pca = PCA(n_components=2)
pca_result = pca.fit_transform(train_features)
plt.figure(figsize=(8, 6))

for label in np.unique(labels):
    plt.scatter(pca_result[labels == label, 0], pca_result[labels == label, 1], label=f'Class {label}', alpha=0.5)

plt.title('PCA Visualization with Labels')
plt.legend()
plt.show()