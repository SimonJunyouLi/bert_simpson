from datetime import datetime
import pandas as pd
import numpy as np
import torch
import csv
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, labels):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.long)
        }

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

user = sys.argv[1]
if user == 'simon':
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_path = '../twitter-datasets/processed_test.csv'
train_path = '../twitter-datasets/processed_train.csv'

train_processed = pd.read_csv(train_path)
tweets = train_processed['text'].values
labels = train_processed['label'].values

test_processed = pd.read_csv(test_path)
test_tweets = test_processed['text'].values
test_ids = test_processed['ids'].values

train_tweets, val_tweets, train_labels, val_labels = train_test_split(tweets, labels, test_size=0.2, random_state=42)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LEN = 128
BATCH_SIZE = 32

train_set = TweetDataset(train_tweets, tokenizer, MAX_LEN, train_labels)
val_set = TweetDataset(val_tweets, tokenizer, MAX_LEN, val_labels)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

EPOCHS = 2
LEARNING_RATE = 2e-5

### OPTIMIZER & SCHEDULER ###
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

train_data_size = len(train_tweets)
steps_per_epoch = int(train_data_size / BATCH_SIZE)
num_train_steps = steps_per_epoch * EPOCHS

# Setting up a polynomial decay schedule
def decay_schedule(step):
    return max(0, (1.0 - step / num_train_steps))

# Setting up a linear warm-up schedule
def warmup_schedule(step):
    return min(1.0, step / (num_train_steps * 0.1))

# Combine the decay and warm-up schedules
def lr_lambda(step):
    return decay_schedule(step) * warmup_schedule(step)

# Defining the AdamWeightDecay optimizer
optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay=0.0)
# Create a LambdaLR scheduler with the combined schedule
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

loss_function = torch.nn.CrossEntropyLoss()

### Training ###
for epoch in range(EPOCHS):
    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
    for batch in train_bar:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = loss_function(outputs.logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_bar.set_postfix(loss=loss.item())

    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    ### Validation ###
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation")
    for batch in val_bar:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)

        with torch.no_grad():
            outputs = model(ids, mask)
            loss = loss_function(outputs.logits, targets)
            total_loss += loss.item()

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            _, predicted = torch.max(probabilities, 1)

            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            val_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples

    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

model.save_pretrained('../model/distilbert_' + datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '_' + str(accuracy) + '%')