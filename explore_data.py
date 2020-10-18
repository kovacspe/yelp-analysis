from dataset import Dataset, read_json, DATA_FILES, read_numeric_data_from_reviews
import os
import matplotlib.pyplot as plt


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
    numeric_data = read_numeric_data_from_reviews(
        DATA_FILES['review'])
    numeric_data['useful'] = numeric_data.apply(
        lambda row: row['useful'] if row['useful'] < 5 else 5, axis=1)
    print(numeric_data[['stars', 'useful']])
    numeric_data_agg = numeric_data[['stars', 'useful']].groupby(
        ['stars', 'useful']).size().reset_index(name='Count')

    # Compute means
    useful_mean = numeric_data['useful'].mean()
    stars_mean = numeric_data['stars'].mean()

    # Compute MSE when always predicting mean
    numeric_data['stars_se'] = numeric_data.apply(
        lambda row: (row['stars']-stars_mean)**2, axis=1)
    numeric_data['useful_se'] = numeric_data.apply(
        lambda row: (row['useful']-useful_mean)**2, axis=1)
    stars_mse = numeric_data['stars_se'].mean()
    useful_mse = numeric_data['useful_se'].mean()

    # Print results
    print(f'Useful mean:{useful_mean}. MSE when predicting mean: {useful_mse}')
    print(f'Stars mean:{stars_mean}. MSE when predicting mean: {stars_mse}')
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


explore_reviews()
