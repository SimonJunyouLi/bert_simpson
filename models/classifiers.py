import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

import sys
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from keras.models import Sequential
from keras.layers import Dense

sys.path.insert(0, os.path.dirname(os.getcwd()))

import helpers.preprocessing_helper_v2 as preprocessing
import helpers.classifiers_helper as classifier
import helpers.submission_helper as sub

clean_data_again = True
debug = True

# select if we want to order the vocabulary by frequency in the text [needed for the calculation of the weight]
reorder_vocabulary = True
# Select if we want to get the list of the words present in the text but not in the vocabulary
list_words_not_in_vocab = False

pooling_method = "mean" # "mean", "max", "tfidf", "weight"
model_type = "logistic" # "logistic", "svm", "neural_net"

if debug and clean_data_again:
  #clean_data_again = False
  print("Warning: debug mode is on and clean_data_again has been reset to False.")
  
if not reorder_vocabulary and pooling_method == "weight":
  reorder_vocabulary = True
  print("Warning: reorder_vocabulary has been reset to True. Because pooling_method is set to weight and needs the vocabulary to be ordered by frequency in the text.")

# Create an instance of StandardScaler
scaler = StandardScaler()

# Load data -------------------------------------------------------------------
# Load the word embeddings
word_embeddings = np.load('word_embeddings/embeddings.npy')
df_word_embeddings = pd.DataFrame(word_embeddings)

# Load the test set tweets
with open('../twitter-datasets/test_data.txt', 'r', encoding='utf-8') as file:
    test_tweets = file.readlines()
    df_test_tweets = pd.DataFrame(test_tweets)

# Load the vocabulary
with open('word_embeddings/processed_vocab_cut.txt', 'r', encoding='utf-8') as file:
    vocabulary = file.read().splitlines()

# Load positive training tweets and assign labels
with open('../twitter-datasets/train_pos_full.txt', 'r', encoding='utf-8') as file:
    pos_tweets = file.readlines()

# Load negative training tweets and assign labels
with open('../twitter-datasets/train_neg_full.txt', 'r', encoding='utf-8') as file:
    neg_tweets = file.readlines()

# Clean data ------------------------------------------------------------------
if debug:
    pos_tweets = pos_tweets[:10]
    neg_tweets = neg_tweets[:10]
    test_tweets = test_tweets[:10]
    #all_tweets = np.concatenate((train_tweets, test_tweets), axis=0)
    vocabulary = vocabulary[:100]

# Transform the data to reuse the code from someone else
pos_labels = np.ones(len(pos_tweets), dtype=int)  # Assign label 1 for positive tweets
neg_labels = -1 * np.ones(len(neg_tweets), dtype=int)  # Assign label -1 for negative tweets

pos_tweets = pd.DataFrame({'text': pos_tweets, 'label': pos_labels})
neg_tweets = pd.DataFrame({'text': neg_tweets, 'label': neg_labels})
ids = np.arange(len(test_tweets))+1

test_tweets = pd.DataFrame({'ids' : ids, 'text': test_tweets})

# Use our cleaning function
pos_tweets = preprocessing.classifier_preprocessing(pos_tweets)
pos_tweets = pos_tweets.drop_duplicates(subset=['text'])
neg_tweets = preprocessing.classifier_preprocessing(neg_tweets)
neg_tweets = neg_tweets.drop_duplicates(subset=['text'])
test_tweets = preprocessing.classifier_preprocessing(test_tweets)


# from pd dataframe to np array
pos_labels = np.array(pos_tweets['label'])
neg_labels = np.array(neg_tweets['label'])
pos_tweets = np.array(pos_tweets['text'])
neg_tweets = np.array(neg_tweets['text'])
test_tweets = np.array(test_tweets['text'])

#reorder the vocabulary and the word embeddings according to the largest number of occurences first
if reorder_vocabulary :
    vocabulary, word_to_embeddings = classifier.reorder_vocabulary(pos_tweets, neg_tweets, test_tweets, vocabulary, word_embeddings, clean_data_again, save_counts=True)

# Convert the values into a NumPy array
word_embeddings = np.array(list(word_to_embeddings.values()))

# Create a dictionary to map words to their corresponding embeddings
word_to_embedding = {word: word_embeddings[i] for i, word in enumerate(vocabulary)}

if list_words_not_in_vocab :
    #save the words that are not in the vocabulary
    classifier.out_of_vocab_file(pos_tweets, neg_tweets, test_tweets, vocabulary, clean_data_again)

### TRAINING THE  CLASSIFIER

train_features, test_features = classifier.get_features(pooling_method, pos_tweets, neg_tweets, test_tweets, word_embeddings, vocabulary, clean_data_again)
# Split the data into training and validation sets
labels = np.concatenate((pos_labels, neg_labels), axis=0)

train_features = np.array(train_features)
labels = np.array(labels)
# Assuming train_features and labels are NumPy arrays
assert len(train_features) == len(labels), "Features and labels must be of the same length"

# Generate a permutation of indices
shuffled_indices = np.random.permutation(len(train_features))

# Apply the shuffled indices to both features and labels
shuffled_features = train_features[shuffled_indices]
shuffled_labels = labels[shuffled_indices]

X_train, X_val, y_train, y_val = train_test_split(train_features, labels, test_size=0.1, random_state=42)

if model_type == "logistic":
    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)


elif model_type == "svm":
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_val)

    # Create an SGDClassifier with a linear SVM loss
    model = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=100, random_state=42, learning_rate='optimal', eta0=0.0, early_stopping=True, n_iter_no_change=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

elif model_type == "neural_net":

    # Create and compile your Keras model
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, epochs=25, batch_size=64, validation_split=0.1)
    
    # Make predictions on your test data
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    
else:
    raise ValueError("model_type is not recognized")

# Validate
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

###  PREDICTIONS

# Construct feature representations for test tweets
test_features = np.array(test_features)
# Make predictions
y_test_pred = model.predict(test_features)

test_data_path = "../twitter-datasets/test_data.txt"
ids_test = sub.get_test_ids(test_data_path)
y_pred = []
y_pred = y_test_pred
y_pred[y_pred <= 0] = -1
y_pred[y_pred > 0] = 1
y_pred = y_pred.astype(int)
print(y_pred)
sub.create_csv_submission(ids_test, y_pred, "../submissions/submission_"+pooling_method+"_"+model_type+".csv")


