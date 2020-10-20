from dataset import Dataset, read_json, DATA_FILES, read_numeric_data_from_reviews, to_one_hot
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd


def load_business():
    path = '.\\data'
    business = read_json(DATA_FILES['business'])

    print(f'Business columns:{business.columns}')
    print(business['categories'])
    business['categories'] = business.apply(
        lambda row: row['categories'].split(', ') if row['categories'] else [], axis=1)
    print(business['categories'].explode().value_counts())
    x = business['categories'].explode().value_counts().iloc[0:50]
    print(x)


def load_users():
    path = '.\\data'
    user = read_json(DATA_FILES['user'], max_lines=100)
    print(user.head())
    print(f'Columns:{user.columns}')


def load_reviews():
    path = '.\\data'
    reviews = read_json(DATA_FILES['review'], max_lines=100)
    print(reviews.head())
    print(f'Columns:{reviews.columns}')


def explore_sentiment(sentiment_stats_file):
    data = pd.read_csv(sentiment_stats_file)
    # Filter out rare occurances
    data = data[data['count'] > 1000]
    data = data[data['std'] < 1]
    # Sort by std
    data.sort_values(by=['mean'], ascending=True, inplace=True)
    # Filter out neutral words
    pos_data = data[data['mean'] > 4.25]
    neg_data = data[data['mean'] < 1.75]
    print(pos_data.head(30))
    print(neg_data.head(30))


def explore_reviews():
    # Load data
    numeric_data = read_numeric_data_from_reviews(
        DATA_FILES['review'], max_lines=1000000)
    # clip useful attribute to interval 0-5
    numeric_data['useful_clipped'] = numeric_data.apply(
        lambda row: row['useful'] if row['useful'] < 30 else 30, axis=1)
    numeric_data = numeric_data[numeric_data['useful'] >= 0]

    # Aggregate data to get pivots
    numeric_data_agg = numeric_data[['stars', 'useful']].groupby(
        ['stars', 'useful']).size().reset_index(name='Count')

    # Compute means and most common features
    useful_mean = numeric_data['useful_clipped'].mean()
    most_common_useful = np.argmax(np.bincount(numeric_data['useful']))
    stars_mean = numeric_data['stars'].mean()
    most_common_stars = np.argmax(np.bincount(numeric_data['stars']))

    # Encode to one-hot fomrat
    useful = to_one_hot(numeric_data['useful'])
    stars = to_one_hot(numeric_data['stars'])

    # Compute sparse categorical corssentropy
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    useful_scce = scce(np.full((len(numeric_data)),
                               most_common_useful), useful).numpy()
    stars_scce = scce(np.full((len(numeric_data)),
                              most_common_stars-1), stars).numpy()

    # Compute MSE when always predicting mean
    numeric_data['stars_se'] = numeric_data.apply(
        lambda row: (row['stars']-stars_mean)**2, axis=1)
    numeric_data['useful_se'] = numeric_data.apply(
        lambda row: (row['useful']-useful_mean)**2, axis=1)
    stars_mse = numeric_data['stars_se'].mean()
    useful_mse = numeric_data['useful_se'].mean()

    # Print results
    print(
        f'Useful mean:{useful_mean}. Baseline MSE: {useful_mse}. Baseline crossentropy: {useful_scce}')
    print(
        f'Stars mean:{stars_mean}. Baseline MSE: {stars_mse}. Baseline crossentropy: {stars_scce}')
    print(numeric_data_agg)

    # classify

    def classify(num_stars):
        if num_stars < 3:
            return 1
        elif num_stars == 3:
            return 3
        else:
            return 5
    numeric_data['class'] = numeric_data.apply(
        lambda row: classify(row['stars']), axis=1)

    counts, _ = np.histogram(numeric_data['class'], bins=[1, 3, 5, 5])
    print(counts/len(numeric_data))
    print(counts[[0, 2]]/np.sum(counts[[0, 2]]))
    plt.title('Distribution of positive and negative reviews')
    plt.bar(['neg', 'neut', 'pos'], height=counts/len(numeric_data))
    plt.show()

    useful_only = numeric_data[numeric_data['useful'] >= 30]
    plt.title('Distribution of stars in useful reviews vs all reviews')
    plt.hist([useful_only['stars'], numeric_data['stars']],
             density=True, bins=5, label=['useful reviews', 'all reviews'])
    plt.legend(loc='upper left')
    plt.show()

    plt.title('Distribution of votes useful in all reviews')
    plt.hist(numeric_data['useful'], bins=200)
    plt.show()

    numeric_data['useful_clipped'] = numeric_data.apply(
        lambda row: row['useful'] if row['useful'] < 5 else 5, axis=1)
    plt.hist2d(numeric_data['stars'],
               numeric_data['useful_clipped'], bins=(5, 5))
    plt.xlabel('stars')
    plt.ylabel('usefull')
    plt.show()


path = os.path.join(
    'data', 'regex_tokens_without_stop_words', 'sentiment.csv')
explore_reviews()
