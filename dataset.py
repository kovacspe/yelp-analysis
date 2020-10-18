import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
import statistics as st
import matplotlib.pyplot as plt

DATA_FILES = {
    'business': os.path.join('data', 'yelp_academic_dataset_business.json'),
    'checkin': os.path.join('data', 'yelp_academic_dataset_checkin.json'),
    'review': os.path.join('data', 'yelp_academic_dataset_review.json'),
    'tip': os.path.join('data', 'yelp_academic_dataset_tip.json'),
    'user': os.path.join('data', 'yelp_academic_dataset_user.json')
}


def read_json(file, max_lines=None):
    list_it = []
    with open(file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            list_it.append(json.loads(line))
            if max_lines:
                if i >= max_lines:
                    break
    return pd.DataFrame(list_it)


def read_numeric_data_from_reviews(file, max_lines=None):
    list_it = []
    with open(file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            json_line = json.loads(line)
            numeric_data = {x: json_line[x]
                            for x in ['stars', 'useful', 'funny', 'cool']}
            list_it.append(numeric_data)
            if max_lines:
                if i >= max_lines:
                    break
    return pd.DataFrame(list_it)


class Dataset:
    def __init__(self, path):
        self.data_path = path

    def load_all(self):
        self.data = read_json(self.data_path)

    def batches(self, batch_size):
        list_it = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                list_it.append(json.loads(line))
                if (i+1) % batch_size == 0:
                    yield list_it
                    list_it = []


class ReviewDataset(Dataset):
    def __init__(self, path, test_ratio):
        super(ReviewDataset, self).__init__(path)
        self.load_all(test_ratio)

    def load_all(self, test_ratio):
        with open(self.data_path, 'rb') as f:
            self.inp = np.load(f, allow_pickle=True)
            self.inp = tf.keras.preprocessing.sequence.pad_sequences(
                self.inp, maxlen=128, padding='post', truncating='post'
            )
            self.stars = np.load(f)
            self.useful = np.load(f)
            self.num_data = len(self.stars)
            self.train_idx = np.arange(
                int(self.num_data*(1-test_ratio)), dtype=np.int32)
            self.test_idx = np.arange(
                int(self.num_data*(1-test_ratio)), self.num_data, dtype=np.int32)

    def find_out_num_words(self):
        maximum = 0
        for x in range(self.num_data):
            m = max(self.inp[x])
            if m > maximum:
                maximum = m
        self.num_words = maximum
        return self.num_words

    def batches(self, batch_size, train=True, num_batches=None):
        if train:
            order = self.train_idx
        else:
            order = self.test_idx
        np.random.shuffle(order)
        for i in range(0, self.num_data, batch_size):
            ind = order[i:i+batch_size]
            yield self.inp[ind], self.stars[ind], self.useful[ind]
            if num_batches and num_batches < i/batch_size:
                break