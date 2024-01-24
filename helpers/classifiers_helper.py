import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import csv
import pandas as pd
import multiprocessing

### POOLING FUNCTIONS

def average_word_vectors(tweet, word_to_embedding):
    # tweet is the sentence
    # words are the words in the sentence
    # word_to_embedding is the dictionary mapping words to their embeddings
    words = tweet.split()
    vectors = []
    for word in words:
        if word in word_to_embedding:
            vectors.append(word_to_embedding[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # If none of the words in the tweet are in the embeddings, return a zero vector
        return np.zeros_like(word_to_embedding[0])

def max_word_vectors(tweet, word_to_embedding):
    # tweet is the sentence
    # words are the words in the sentence
    # word_to_embedding is the dictionary mapping words to their embeddings
    words = tweet.split()
    vectors = []
    for word in words:
        if word in word_to_embedding:
            vectors.append(word_to_embedding[word])
    if vectors:
        return np.max(vectors, axis=0)
    else:
        # If none of the words in the tweet are in the embeddings, return a zero vector
        return np.zeros_like(word_to_embedding[0])


def weighted_average_word_vectors(tweet, word_to_embedding, weight_, vocabulary):
    words = tweet.split()
    vectors = []
    weights = []

    for word in words:
        if word in word_to_embedding:
            vectors.append(word_to_embedding[word])
        else:
            zero_vector = np.zeros_like(word_to_embedding[0])
            vectors.append(zero_vector)

    for word in words:
        if word in vocabulary:
            word_idx = vocabulary.index(word)
            weights.append(weight_[word_idx])
        else:
            weights.append(0)

    weights = np.array(weights)


    if np.any(weights):  # Check if all weights are not zero
        weights /= np.sum(weights)
    else:
        weights = np.ones_like(weights) / len(weights)

    return np.average(vectors, axis=0, weights=weights) # Return the weighted average


def get_features(pooling_method, train_tweets_pos, train_tweets_neg, test_tweets, word_to_embedding, vocabulary, clean_data_again):
    if pooling_method == "mean" :
        train_tweets = np.concatenate((train_tweets_pos, train_tweets_neg), axis=0)
        # Construct feature representations for training tweets
        train_features = [average_word_vectors(tweet, word_to_embedding) for tweet in train_tweets]
        test_features = [average_word_vectors(tweet, word_to_embedding) for tweet in test_tweets]
        
    elif pooling_method == "max" :
        train_tweets = np.concatenate((train_tweets_pos, train_tweets_neg))

        train_features = [max_word_vectors(tweet, word_to_embedding) for tweet in train_tweets]
        test_features = [max_word_vectors(tweet, word_to_embedding) for tweet in test_tweets]

    elif pooling_method == "tfidf":
        train_tweets = np.concatenate((train_tweets_pos, train_tweets_neg))

        all_features = []
        all_tweets = np.concatenate((train_tweets, test_tweets))
        
        # Create the vectorizer
        vectorizer = TfidfVectorizer(vocabulary=vocabulary)

        # Fit and transform the tweets
        tfidf = vectorizer.fit_transform(all_tweets)

        for doc_index, tweet in enumerate(all_tweets):
            tfidf_vector = tfidf[doc_index].todense().A1  # Convert to dense format and flatten
            feature = weighted_average_word_vectors(tweet, word_to_embedding, tfidf_vector, vocabulary)
            all_features.append(feature)

        train_features = all_features[:len(train_tweets)]
        test_features = all_features[len(train_tweets):]
        
    elif pooling_method == "weight":
        weights = calculate_weights(train_tweets_pos, train_tweets_neg, vocabulary, clean_data_again)
        train_tweets = np.concatenate((train_tweets_pos, train_tweets_neg))
        all_features = []
        all_tweets = np.concatenate((train_tweets, test_tweets))
        
        for _, tweet in enumerate(all_tweets):
            feature = weighted_average_word_vectors(tweet, word_to_embedding, weights, vocabulary)
            all_features.append(feature)
            
        train_features = all_features[:len(train_tweets)]
        test_features = all_features[len(train_tweets):]
        
            
    else : 
        raise ValueError("Pooling method not recognized")
            
    return train_features, test_features

### VOCAB MANIP FUNCTIONS

def remove_single_and_double_letters(train_tweets, test_tweets):
    cleaned_tweets_train = []
    for tweet in train_tweets:
        # Split the tweet into words and filter out single-letter words
        words = tweet.split()
        cleaned_tweet = ' '.join(word for word in words if len(word) > 2)
        cleaned_tweets_train.append(cleaned_tweet)
        
    cleaned_tweets_test = []
    for tweet in test_tweets:
        # Split the tweet into words and filter out single-letter words
        words = tweet.split()
        cleaned_tweet = ' '.join(word for word in words if len(word) > 2)
        cleaned_tweets_test.append(cleaned_tweet)
    return cleaned_tweets_train, cleaned_tweets_test

def save_to_csv_vocabulary(sorted_vocabulary, counts, filename):
    data = zip(sorted_vocabulary, counts)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Count'])  # Write header row
        for row in data:
            writer.writerow(row)
            
            
def reorder_vocabulary(pos_tweets, neg_tweets, test_tweets, vocabulary, word_embeddings, clean_data_again, save_counts):
    if clean_data_again : 
        train_tweets = np.concatenate((pos_tweets, neg_tweets), axis=0)
        text_list = np.concatenate((train_tweets, test_tweets), axis=0)
        # Initialize a dictionary to count occurrences of each word
        word_counts = {word: 0 for word in vocabulary}

        # Count occurrences of each word in the vocabulary
        for text in text_list:
            words = text.split()  # Split text into words
            for word in words:
                if word in word_counts:
                    word_counts[word] += 1

        # Sort the vocabulary based on the count, in descending order
        sorted_vocabulary = sorted(word_counts, key=word_counts.get, reverse=True)

        # Create a list of word counts in the same order as the sorted vocabulary
        counts = [word_counts[word] for word in sorted_vocabulary]

        # Update word_embeddings to match the new order of words
        new_word_to_embedding = {word: word_embeddings[vocabulary.index(word)] for word in sorted_vocabulary}
        
        # Save sorted_vocabulary
        with open('../processed_data/sorted_vocabulary_by_occurences.txt', 'w', encoding='utf-8') as file:
            for word in sorted_vocabulary:
                file.write(word + '\n')

        # Save counts
        with open('../processed_data/words_counts_sorted_from_vocab.txt', 'w', encoding='utf-8') as file:
            for count in counts:
                file.write(str(count) + '\n')

        # Save new_word_to_embedding to a .txt file
        with open('../processed_data/word_embeddings_sorted_by_occurences.txt', 'w', encoding='utf-8') as file:
            for word, embedding in new_word_to_embedding.items():
                embedding_str = ' '.join(map(str, embedding))
                file.write(f"{word} {embedding_str}\n")

    else :
        
        # Reload sorted_vocabulary
        with open('../processed_data/sorted_vocabulary_by_occurences.txt', 'r', encoding='utf-8') as file:
            sorted_vocabulary = [line.strip() for line in file]

        # Reload counts
        with open('../processed_data/words_counts_sorted_from_vocab.txt', 'r', encoding='utf-8') as file:
            counts = [int(line.strip()) for line in file]

        # Load new_word_to_embedding from a .txt file
        new_word_to_embedding = {}
        with open('../processed_data/word_embeddings_sorted_by_occurences.txt', 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                word = parts[0]
                embedding = [float(val) for val in parts[1:]]
                new_word_to_embedding[word] = embedding



        
    if save_counts:
        save_to_csv_vocabulary(sorted_vocabulary, counts, '../processed_data/vocabulary_counts.csv')

    return sorted_vocabulary, new_word_to_embedding


def calculate_weights(text_pos, text_neg , vocabulary, clean_data_again):
    if clean_data_again:
        # Initialize a dictionary to count occurrences of each word
        word_counts_pos = {word: 0 for word in vocabulary}
        word_counts_neg = {word: 0 for word in vocabulary}
        word_ratio = {word: 0 for word in vocabulary}
        
        # Count occurrences of each word in the vocabulary
        for text in text_pos:
            words = text.split()  # Split text into words
            for word in words:
                if word in word_counts_pos:
                    word_counts_pos[word] += 1
                    
        for text in text_neg:
            words = text.split()  # Split text into words
            for word in words:
                if word in word_counts_neg:
                    word_counts_neg[word] += 1
                    
        for word in vocabulary:
            count_pos = word_counts_pos[word]
            count_neg = word_counts_neg[word]

            if (count_neg < count_pos) :
                if count_neg == 0:
                    count_neg = 1
                ratio = count_pos / count_neg
            else : 
                if count_pos == 0:
                    count_pos = 1
                ratio = count_neg / count_pos

            word_ratio[word] = ratio
            
        # Convert dictionary to a numpy array and scale
        ratios_array = np.array(list(word_ratio.values()))
        scaled_ratios = (ratios_array) / (np.max(ratios_array)) #ratio was between 1 and infty, now between 0 and 1

        # Update the dictionary with scaled values for saving
        for i, word in enumerate(vocabulary):
            word_ratio[word] = scaled_ratios[i]

            
        # Save the scaled dictionary to a Pandas DataFrame with 'word' as the index
        df_word_ratio = pd.DataFrame({'word': list(word_ratio.keys()), 'ratio': list(word_ratio.values())})
        df_word_ratio = df_word_ratio.set_index('word')

        # Save the DataFrame to a text file
        df_word_ratio.to_csv('../processed_data/word_ratios_descending.csv')



                
        np.savetxt("../processed_data/word_ratios_descending.txt", scaled_ratios)
    
    else:
        scaled_ratios = np.loadtxt("../processed_data/word_ratios_descending.txt")

    return scaled_ratios



def find_out_of_vocab_words(tweets, vocabulary):
    # Find and return out-of-vocabulary words in the given tweets
    out_of_vocab_words = set(word for tweet in tweets for word in tweet.split() if word not in vocabulary)
    return out_of_vocab_words

def out_of_vocab_file(pos_tweets, neg_tweets, test_tweets, vocabulary, clean_data_again):
    if clean_data_again:
        # Create a multiprocessing pool with 3 workers
        with multiprocessing.Pool(3) as pool:
            # Map each tweet dataset to its processor
            results = pool.starmap(find_out_of_vocab_words, [
                (pos_tweets, vocabulary),
                (neg_tweets, vocabulary),
                (test_tweets, vocabulary)
            ])

        # Combine results from all three processors
        combined_out_of_vocab_words = set().union(*results)

        # Write the combined unique words to a file
        with open('../processed_data/out_of_vocab_words.txt', 'w', encoding='utf-8') as file:
            file.writelines(word + '\n' for word in combined_out_of_vocab_words)