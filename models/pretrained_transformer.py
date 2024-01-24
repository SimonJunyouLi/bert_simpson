# import datasets
import numpy as np
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import pipeline
from transformers.pipelines.base import KeyDataset
import tqdm

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def get_test_ids(path):
    file = open(path,'r')
    lines = file.readlines()
    for rowidx in range(len(lines)):
        index = lines[rowidx].index(',')
        lines[rowidx] = lines[rowidx][:index]
    return lines

test_data_path = '../twitter-datasets/processed_test_data.txt'
train_pos_path = '../twitter-datasets/processed_train_pos.txt'
train_neg_path = '../twitter-datasets/processed_train_neg.txt'

# Load the test set tweets
with open(test_data_path, 'r', encoding='utf-8') as file:
    test_tweets = file.readlines()

# Load positive training tweets and assign labels
with open(train_pos_path, 'r', encoding='utf-8') as file:
    pos_tweets = file.readlines()

pos_labels = np.ones(len(pos_tweets), dtype=int) 

# Load negative training tweets and assign labels
with open(train_neg_path, 'r', encoding='utf-8') as file:
    neg_tweets = file.readlines()

neg_labels = 0 * np.ones(len(neg_tweets), dtype=int)  

tweets = test_tweets
labels = np.concatenate((pos_labels, neg_labels), axis=0)


pipe = pipeline("sentiment-analysis")
result = []
for tweet in tweets:
    result.append(pipe(tweet)[0]['label'])

result = [1 if x == 'POSITIVE' else -1 for x in result]

ids_test = get_test_ids('../twitter-datasets/test_data.txt')
create_csv_submission(ids_test, result, "../submissions/submission_trans.csv")

