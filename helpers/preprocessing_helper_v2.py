import re
import pkg_resources
import numpy as np
import pandas as pd
import nltk
import ssl
import sys
import multiprocessing
import subprocess

from symspellpy import SymSpell
from collections import deque
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

symspell = SymSpell()

dictionary_path = pkg_resources.resource_filename(
'symspellpy',
'frequency_dictionary_en_82_765.txt')

symspell.load_dictionary(dictionary_path, term_index=0,
                                        count_index=1)

bigram_path = pkg_resources.resource_filename(
'symspellpy',
'frequency_bigramdictionary_en_243_342.txt')

symspell.load_bigram_dictionary(bigram_path, term_index=0,
                                            count_index=2)

file_path = ['../twitter-datasets/train_neg.txt', '../twitter-datasets/train_pos.txt']
full_file_path = ['../twitter-datasets/train_neg_full.txt', '../twitter-datasets/train_pos_full.txt']
test_file_path = ['../twitter-datasets/test_data.txt']

def get_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def drop_duplicates():
    data = data.drop_duplicates(subset=['text'])
    
def remove_elongs():
    data['text'] = data['text'].apply(
      lambda text: str(re.sub(r'\b(\S*?)(.)\2{3,}\b', r'\1\2\2\2', text)))

def lower_case():
    data['text'] = data['text'].str.lower()

def spell_correct():
    data['text'] = data['text'].apply(lambda text: symspell.lookup_compound(text, max_edit_distance=2)[0].term)

def lemmatize(text):
    nltk_tagged = nltk.pos_tag(text.split())
    lemmatizer = WordNetLemmatizer()

    return ' '.join(
      [lemmatizer.lemmatize(w, get_wordnet_tag(nltk_tag))
       for w, nltk_tag in nltk_tagged])

def lemmatizer():
    data['text'] = data['text'].apply(lemmatize)

def stopword():
    stopwords_ = set(stopwords.words('english'))

    data['text'] = data['text'].apply(
      lambda text: ' '.join(
        [word for word in str(text).split() if word not in stopwords_]))
    
def hashtag():
    data['text'] = data['text'].apply(
      lambda text: str(re.sub(r'[\<].*?[\>]', '', text)))
    data['text'] = data['text'].apply(lambda text: text.strip())

def split_hashtag(text, token='#', vocabulary=set()):
    # Find the word(s) following the token
    match = re.search(f"{token}(\w+)", text)
    if not match:
        return []

    hashtag = match.group(1)
    
    # Check if the entire hashtag is in the vocabulary
    full_hashtag = token + hashtag
    if full_hashtag in vocabulary:
        return [full_hashtag]
    
    words = []
    remaining = deque(hashtag)

    while remaining:
        current = ''
        best_match = None

        for char in list(remaining):
            current += char
            if current in vocabulary:
                # If current match is earlier in the vocabulary, update best_match
                if best_match is None or vocabulary.index(current) < vocabulary.index(best_match):
                    best_match = current

        if best_match:
            words.append(best_match)
            for _ in range(len(best_match)):
                remaining.popleft()  # Remove the matched part from remaining
        else:
            break  # No further matches found

    return words if words else [full_hashtag]  # Return the original hashtag if no words found

def remove_hashtags(train_tweets, test_tweets, vocabulary):
    # Remove '#' characters from each tweet in the list
    train_tweets = [tweet.replace("#", "") for tweet in train_tweets]
    test_tweets = [tweet.replace("#", "") for tweet in test_tweets]
    vocabulary = [word.replace("#", "") for word in vocabulary]    
    return

def process_tweet(tweet, vocabulary):
    hashtag_regex = re.compile(r"#\w+")
    hashtags = hashtag_regex.findall(tweet)
    processed_tweet = tweet

    for hashtag in hashtags:
        split_words = split_hashtag(hashtag, vocabulary=vocabulary)
        processed_tweet = processed_tweet.replace(hashtag, ' '.join(split_words))

    return processed_tweet

def process_tweets_hashtags_parallel(tweets, vocabulary):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        processed_tweets = pool.starmap(process_tweet, [(tweet, vocabulary) for tweet in tweets])

    return processed_tweets

def process_tweets_hashtags(train_tweets_pos, train_tweets_neg, test_tweets, vocabulary, clean_data_again):
    if clean_data_again:
        processed_train_tweets_pos = process_tweets_hashtags_parallel(train_tweets_pos, vocabulary)
        processed_train_tweets_neg = process_tweets_hashtags_parallel(train_tweets_neg, vocabulary)
        processed_test_tweets = process_tweets_hashtags_parallel(test_tweets, vocabulary)

        # Write the contents into files
               # Write the contents into a new file
        with open('../processed_data/process_tweets_hashtags_train_pos.txt', 'w', encoding='utf-8') as file:
            file.writelines(processed_train_tweets_pos)
            
        with open('../processed_data/process_tweets_hashtags_train_neg.txt', 'w', encoding='utf-8') as file:
            file.writelines(processed_train_tweets_neg)
            
        with open('../processed_data/process_tweets_hashtags_test_data.txt', 'w', encoding='utf-8') as file:
            file.writelines(processed_test_tweets)

    else:
        # Load the processed data
        with open('../processed_data/process_tweets_hashtags_train_pos.txt', 'r', encoding='utf-8') as file:
            processed_train_tweets_pos = file.readlines()
            
        with open('../processed_data/process_tweets_hashtags_train_neg.txt', 'r', encoding='utf-8') as file:
            processed_train_tweets_neg = file.readlines()
            
        with open('../processed_data/process_tweets_hashtags_test_data.txt', 'r', encoding='utf-8') as file:
            processed_test_tweets = file.readlines()

    return processed_train_tweets_pos, processed_train_tweets_neg, processed_test_tweets

def remove_tags():
    # data['text'] = data['text'].apply(
    #   lambda text: str(re.sub(r'[\<].*?[\>]', '', text)))
    data.replace(r'<.*?>', '', regex=True, inplace=True)
    data['text'] = data['text'].apply(lambda text: text.strip())
    data['text'] = data['text'].str.replace('\.{3}$', '')

def filter_alpha(tokens):
    return [word for word in tokens if word.isalpha()]

def letters():
    data['text'] = data['text'].apply(lambda text: filter_alpha(text.split()))

def prune_punctuations():
    data['text'] = data['text'].replace({'[$&+=@#|<>:*()%]': ''}, regex=True)

def empty():
    # data['text'] = data['text'].str.replace('^\s*$', '<EMPTY>')
    data.replace("", "<EMPTY>", inplace=True)

def spacing():
    # rewrite
    data['text'] = data['text'].str.replace('\s{2,}', ' ')
    data['text'] = data['text'].apply(lambda text: text.strip())
    data.reset_index(inplace=True, drop=True)

def final_parenthesis(self):
    data['text'] = data['text'].str.replace('\)+$', ':)')
    data['text'] = data['text'].str.replace('\(+$', ':(')

def preprocess(model, dataset):
    global data 
    data = pd.DataFrame(columns=['text', 'label'])

    if dataset == 'train':
        list = file_path
    elif dataset == 'train_full':
        list = full_file_path
    elif dataset == 'test':
        list = test_file_path
    else:
        list = ['../twitter-datasets/testing.txt']

    if dataset == 'test':
        with open(test_file_path[0]) as f:
            content = f.read().splitlines()
        ids = [line.split(',')[0] for line in content]
        texts = [','.join(line.split(',')[1:]) for line in content]
        data = pd.DataFrame(columns=['ids', 'text'],
                                    data={'ids': ids, 'text': texts})
    else:
        for i, path in enumerate(list):
            with open(path) as f:
                content = f.readlines()

                df = pd.DataFrame(columns=['text', 'label'],
                                data={'text': content,
                                    'label': np.ones(len(content)) * i})

                data = pd.concat([data, df], ignore_index=True)
        
    if dataset == 'train' or dataset == 'train_full':
        data = data.drop_duplicates(subset=['text'])

    if model in ['distilbert', 'bert']:
        lower_case()
        remove_tags()
        final_parenthesis()
        spacing()
        empty()
    elif model in ['logistic', 'svm', 'neural_net']:
        lower_case()
        remove_elongs()
        spell_correct()
        spacing()
        letters()
        lemmatizer()
        stopword()
        hashtag()
        empty()
        spacing()

    data = data.sample(frac=1)

    if dataset == 'train':
        data.to_csv('../twitter-datasets/processed_train.csv', index=False)
    elif dataset == 'train_full':
        data.to_csv('../twitter-datasets/processed_train_full.csv', index=False)
    elif dataset == 'test':
        data.to_csv('../twitter-datasets/processed_test.csv', index=False)
    else: print(data)

    return data

def classifier_preprocessing(target):
    global data 
    data = target        
    lower_case()
    remove_elongs()
    spell_correct()
    spacing()
    letters()
    data['text'] = data['text'].apply(' '.join)
    lemmatizer()
    stopword()
    hashtag()
    empty()
    spacing()
    return data