# CS-433 Project 2 : Tweet Sentiment Classification

## About
Tweet text classification project for positive and negative sentiment as part of the 2023 EPFL Machine Learning Course. Our team name on AIcrowd is [bert_simpson](https://www.aicrowd.com/challenges/epfl-ml-text-classification/teams/bert_simpson). The project is a binary classification task. It aims to determine whether a tweet expresses some positive or some negative sentiment. Several supervised methods have been implemented. A logistic regression, a support vector machine, and a neural network. All these methods were outperformed by BERT models that were also implemented. We used a classical BERT model, a distilled version, and a version specialized for english tweet sentiment analysis.

Tasks of the code for the standard classifiers (logistic regression, support vector machines, neural network):
- Load the data
- Preprocess the data: set all cases to lower cases, remove words that are too long (i.e. "hellooo" becomes "hello"), correct the spelling of all words, check for the correct spacing between words, strip away all non-alphabetic and space character, lemmatize all words, remove stop words (i.e. "the", "in", "and"), remove hashtags, replaces any empty sequence with a special token, recheck for the correct spacing between words
- Optional: reorder the vocabulary according to the word frequency in the tweet sample and save it in a file.
- Optional: list all the words in the tweet not present in the vocabulary
- Train the selected model: either a logistic regression, a support vector machine, or a neural network.
  - All these models can be trained on a sequence embedding that can be calculated:
    - "mean" : Using the mean of the word embedding of the remaining tweet
    - "max" : Using the maximum value for each component in the word embeddings of the tweet.
    - "tfidf" : Using TF-IDF weights for each tweet, where each document is a tweet.
    - "weight" : Using defined weights. It is based on the following algorithm: it calculates the number of occurrences of a word in the positive tweets and the number of occurrences in the negative tweets. If a word is more present in the positive tweets, the weight is set to $\frac{occurence_{pos}}{occurence_{neg}}$, and if a word is more present in the negative tweets, the weight is set to $\frac{occurence_{neg}}{occurence_{pos}}$.

Tasks of the code for BERT (standard BERT, DistilBERT, BERTweet):
- Load the Data
- Preprocess the data according to the basic or advanced preprocessing functions in the report.
- Run the training mode: train specific BERT models such as DistilBERT, BERT or BERTweet.
- Run the inference mode: make predictions on new data using the trained BERT model.

## Setup

1. Required environment :
  This project relies on the following Python libraries:
  
  - numpy (version >= 1.23.5)
  - scikit-learn (version >=  1.2.2)
  - pandas (version >= 2.0.1)
  - keras (version >= 2.15.0)
  - tensorflow (version >= 2.15.0)
  - setuptools (version >= 67.7.2)
  - nltk (version >= 3.8.1)
  - symspellpy (version >= 6.7.7)
  - torch (version >= 2.1.2)
  - setuptools_rust (version >= 1.8.1)
  - transformers (version >= 4.36.2)

    You can install these libraries using `pip` or any other package manager.
    Remark: older versions of the above packages may work, but have not been tested.

2. Clone the repository. Using :

   git clone https://github.com/pmcintyr/sentiMentaL_tweets.git

3. Create a `twitter-datasets` folder in this directory and extract the contents from the CS-433 Project 2 repository:
   https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files

   There are several CSV files: `twitter-datasets/test_data.txt`, `twitter-datasets/train_neg.txt`, `twitter-datasets/train_neg_full.txt`, `twitter-datasets/train_pos.txt`, `twitter-datasets/train_pos_full.txt`, and `twitter-datasets/sample_submission.csv`.
   
Execute `classifiers.py`. in the “models” directory. Check in the file first few lines that the "debug" is set to False, and the "clean_data_again" is set to true. These are needed to pre-process the data. After that choose your pooling method in the choice list: "mean", "max", "tfidf", "weight" as well as the model type in "logistic", "svm", "neural_net". The code with "debug = False" will run over several hours.

Execute `bert.py` in the "models" directory. You may set arguments to specify 1) `username` can be specified for local setups, otherwise write a placeholder username) 2) `model_name` can be distilbert, bert or bertweet depending on which model you would like to run 3) `mode` can be set to train or inference: you should first run the train mode then proceed with the inference mode.

All other required helper functions are contained in the folder `helpers` and loaded in the different scripts.


## Authors
- Junyou Li : junyou.li@epfl.ch
- Wanchai Grossrieder : wanchai.grossrieder@epfl.ch
- Paul McIntyre : paul.mcintyre@epfl.ch

## Licence
This project is not licensed.

## Acknowledgments
We would like to thank the EPFL's professors of the course CS-433 Machine Learning and the associated teaching assistants for their help and support as well for the EPFL cluster access.
