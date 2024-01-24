import pandas as pd
import numpy as np
import torch
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_test_ids(path):
    file = open(path,'r')
    lines = file.readlines()
    for rowidx in range(len(lines)):
        index = lines[rowidx].index(',')
        lines[rowidx] = lines[rowidx][:index]
    return lines    

def create_csv_submission(ids, y_pred, name):
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

test_path = '../twitter-datasets/processed_test.csv'

test_processed = pd.read_csv(test_path)
test_tweets = test_processed['text'].values
test_ids = test_processed["ids"].values

# # Load pre-trained DistilBERT tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertForSequenceClassification.from_pretrained('/Users/simonli/Desktop/epfl/cs433/project2/sentiMentaL_tweets/distilbert', num_labels=2)  # Assuming binary classification

# Load pre-trained DistilBERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForSequenceClassification.from_pretrained('/Users/simonli/Desktop/epfl/cs433/project2/sentiMentaL_tweets/bert', num_labels=2)

# Tokenize and encode tweets for testing set
test_input_ids = [tokenizer.encode(tweet, add_special_tokens=True, max_length=512) for tweet in test_tweets]
test_input_ids = torch.tensor(pad_sequences(test_input_ids, padding='post', truncating='post', maxlen=None))

#-----------------------------------------------------------------------------------------------------#
### PREDICTION ###

# Set the model to evaluation mode
model.eval()

# Forward pass to get logits
with torch.no_grad():
    outputs = model(test_input_ids)

# Convert logits to probabilities using softmax
probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

# Get the predicted class (index with the maximum probability)
_, predicted_class = torch.max(probabilities, 1)

# Convert the result to a numpy array
predictions = predicted_class.cpu().numpy()

#-----------------------------------------------------------------------------------------------------#
### CREATE CSV SUBMISSION ###

ids_test = get_test_ids('../twitter-datasets/test_data.txt')
y_pred = []
y_pred = predictions
y_pred[y_pred <= 0] = -1
y_pred[y_pred > 0] = 1
create_csv_submission(test_ids, y_pred, "../submissions/submission_bert.csv")