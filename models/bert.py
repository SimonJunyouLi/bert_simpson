from datetime import datetime
import pandas as pd
import numpy as np
import torch
import csv
import sys
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification, AutoModel, AutoTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch import nn
from preprocessing_helper import preprocessing

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

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = pooled_output
    return self.out(output)

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

def train():
    train_processed = pd.read_csv(train_path)
    tweets = train_processed['text'].values
    labels = train_processed['label'].values

    train_tweets, val_tweets, train_labels, val_labels = train_test_split(tweets, labels, test_size=0.2, random_state=42)

    train_set = TweetDataset(train_tweets, tokenizer, MAX_LEN, train_labels)
    val_set = TweetDataset(val_tweets, tokenizer, MAX_LEN, val_labels)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    train_data_size = len(train_tweets)
    steps_per_epoch = int(train_data_size / BATCH_SIZE)
    num_train_steps = steps_per_epoch * EPOCHS

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)  # Using AdamW for weight decay
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_train_steps), num_training_steps=num_train_steps)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_train_steps), num_training_steps=num_train_steps)

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

            if model_name == 'bert' or model_name == 'distilbert':
                loss = loss_function(outputs.logits, targets)
            elif model_name == 'bertweet':
                loss = loss_function(outputs, targets)

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

                if model_name == 'bert' or model_name == 'distilbert':
                    loss = loss_function(outputs.logits, targets)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                    _, predicted = torch.max(probabilities, 1)
                    total_correct += (predicted == targets).sum().item()
                elif model_name == 'bertweet':
                    loss = loss_function(outputs, targets)
                    _, preds = torch.max(outputs, dim=1)
                    total_correct += (preds == targets).sum().item()
                
                total_loss += loss.item()
                total_samples += targets.size(0)

                val_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples

        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # if model_name == 'bert' or model_name == 'distilbert':
    #     model.save_pretrained('../model/' + model_name + '_' + datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '_' + str(accuracy) + '%')
    # elif model_name == 'bertweet':
    #     torch.save(model.state_dict(), '../model/bertweet_best_model_state.bin')

def inference():
    test_processed = pd.read_csv(test_path)
    test_tweets = test_processed['text'].values
    test_ids = test_processed["ids"].values

    test_set = TweetDataset(test_tweets, tokenizer, MAX_LEN, test_ids)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    ### Prediction ###
    predictions = []

    model.eval()
    val_bar = tqdm(test_loader, desc=f"Testing")
    for batch in val_bar:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)

        with torch.no_grad():
            outputs = model(ids, mask)

            if model_name == 'bert' or model_name == 'distilbert':
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                _, predicted = torch.max(probabilities, 1)
                predictions.extend(predicted.cpu().numpy())
            elif model_name == 'bertweet':
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())

    ### Create CSV Submission ###
    predictions = np.array(predictions)

    y_pred = []
    y_pred = predictions
    y_pred[y_pred <= 0] = -1
    y_pred[y_pred > 0] = 1
    create_csv_submission(test_ids, y_pred, submission_file_path)

user = sys.argv[1]
model_name = sys.argv[2]
mode = sys.argv[3]
submission_name = sys.argv[4]

test_path = '../twitter-datasets/processed_test.csv'
train_path = '../twitter-datasets/process_train_full.csv'
model_path = '../model/distilbert'
submission_file_path = '../submissions/' + submission_name + '.csv'

if not os.path.isfile(train_path):
    preprocessing(model_name, 'train_full')

if not os.path.isfile(test_path):
    preprocessing(model_name, 'test')

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-5

device = torch.device("mps" if user == 'simon' else "cuda" if torch.cuda.is_available() else "cpu")

if model_name == 'distilbert':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2) if mode == 'train' else DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)

elif model_name == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2) if mode == 'train' else BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

elif model_name == 'bertweet':
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization = True, use_fast=False)
    model = SentimentClassifier(2)

model.to(device)

if mode == 'train': 
    train()
    inference()
elif mode == 'inference': 
    inference()