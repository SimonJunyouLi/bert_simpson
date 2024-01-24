from datetime import datetime
import pandas as pd
import numpy as np
import torch
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
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
train_path = '../twitter-datasets/processed_train.csv'

# train_tweets_path = '../twitter-datasets/text.txt'
# train_labels_path = '../twitter-datasets/label.txt'

# with open(train_tweets_path, 'r', encoding='utf-8') as file:
#     tweets = file.readlines()

# # Load positive training tweets and assign labels
# with open(train_labels_path, 'r', encoding='utf-8') as file:
#     labels = file.readlines()

train_processed = pd.read_csv(train_path)
tweets = train_processed['text'].values
labels = train_processed['label'].values

test_processed = pd.read_csv(test_path)
test_tweets = test_processed['text'].values
test_ids = test_processed['ids'].values

# labels = np.array([float(value.strip()) for value in labels if value.strip() != ''], dtype=np.float64)
# labels = labels.astype(int)

# Split the data into training and testing sets
train_tweets, val_tweets, train_labels, val_labels = train_test_split(tweets, labels, test_size=0.2, random_state=42)

# Load pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', max_length=128, truncation=True)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)  # Assuming binary classification

# Tokenize and encode tweets for training set   
train_input_ids = [tokenizer.encode(tweet, add_special_tokens=True, max_length=128, truncation=True) for tweet in train_tweets]
train_input_ids = torch.tensor(pad_sequences(train_input_ids, padding='post', truncating='post', maxlen=None))
train_labels = torch.tensor(train_labels, dtype=torch.long)

# Tokenize and encode tweets for validation set
val_input_ids = [tokenizer.encode(tweet, add_special_tokens=True, max_length=128, truncation=True) for tweet in val_tweets]
val_input_ids = torch.tensor(pad_sequences(val_input_ids, padding='post', truncating='post', maxlen=None))
val_labels = torch.tensor(val_labels, dtype=torch.long)

# Tokenize and encode tweets for testing set
test_input_ids = [tokenizer.encode(tweet, add_special_tokens=True, max_length=128, truncation=True) for tweet in test_tweets]
test_input_ids = torch.tensor(pad_sequences(test_input_ids, padding='post', truncating='post', maxlen=None))

# Create a DataLoader
dataset = TensorDataset(train_input_ids, torch.tensor(train_labels))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set up optimizer and loss function
# optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

print("start fine-tune")

# Fine-tune the model
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_input_ids = train_input_ids.to(device)
train_labels = train_labels.to(device)

for epoch in range(epochs):
    print(epoch)
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # outputs = model(inputs, labels=labels)
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()

model.save_pretrained('../model/distilbert' + datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))

#-----------------------------------------------------------------------------------------------------#
## VALIDATION & PREDICTIONS ###
val_dataset = TensorDataset(val_input_ids, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

val_input_ids = val_input_ids.to(device)
val_labels = val_labels.to(device)

# Set the model to evaluation mode
model.eval()

# Lists to store predictions
all_predictions = []

# Iterate through the test dataloader
with torch.no_grad():
    for batch in val_dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass to get logits
        logits = model(inputs)

        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits.logits, dim=1)

        # Get the predicted class (index with the maximum probability)
        _, predicted_class = torch.max(probabilities, 1)

        # Append predictions to the list
        all_predictions.extend(predicted_class.cpu().numpy())

# Convert the list to a numpy array
all_predictions = np.array(all_predictions)

# Calculate accuracy
accuracy = accuracy_score(val_labels.cpu().numpy(), all_predictions)
print(f"Validation Accuracy: {accuracy}")

#-----------------------------------------------------------------------------------------------------#
### PREDICTION ###

test_input_ids = test_input_ids.to(device)

# Set the model to evaluation mode
model.eval()

# Forward pass to get logits
with torch.no_grad():
    test_input_ids = test_input_ids.to(device)
    outputs = model(test_input_ids)

# Convert logits to probabilities using softmax
probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

# Get the predicted class (index with the maximum probability)
_, predicted_class = torch.max(probabilities, 1)

# Convert the result to a numpy array
predictions = predicted_class.cpu().numpy()

#-----------------------------------------------------------------------------------------------------#
### CREATE CSV SUBMISSION ###

y_pred = []
y_pred = predictions
y_pred[y_pred <= 0] = -1
y_pred[y_pred > 0] = 1
create_csv_submission(test_ids, y_pred, "../submissions/submission_dbert_20.csv")