from dataset import Dataset, read_json, DATA_FILES, read_numeric_data_from_reviews, to_one_hot
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


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


def explore_reviews():
    # Load data
    numeric_data = read_numeric_data_from_reviews(
        DATA_FILES['review'])
    # clip useful attribute to interval 0-5
    numeric_data['useful'] = numeric_data.apply(
        lambda row: row['useful'] if row['useful'] < 5 else 5, axis=1)
    numeric_data = numeric_data[numeric_data['useful'] >= 0]

    # Aggregate data to get pivots
    numeric_data_agg = numeric_data[['stars', 'useful']].groupby(
        ['stars', 'useful']).size().reset_index(name='Count')

    # Compute means and most common features
    useful_mean = numeric_data['useful'].mean()
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
                              most_common_stars), stars).numpy()

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
    useful_only = numeric_data[numeric_data['useful'] >= 3]

    plt.title('Distribution of stars in useful reviews (useful>3)')
    plt.hist(useful_only['stars'], bins=5)
    plt.show()

    plt.hist2d(numeric_data['stars'], numeric_data['useful'], bins=(5, 5))
    plt.xlabel('stars')
    plt.ylabel('usefull')
    plt.show()
    plt.hist2d(numeric_data['funny'], numeric_data['cool'], bins=(100, 100))
    plt.xlabel('funny')
    plt.ylabel('usefull')
    plt.show()
