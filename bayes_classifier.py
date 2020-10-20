import pandas as pd
import numpy as np
from dataset import Dataset
import os
import sklearn.metrics as metrics


class BayesClassifier:
    def __init__(self, pos_dict, neg_dict, positive_case_rate):
        self.positive_case_rate = np.log(positive_case_rate)
        self.negative_case_rate = np.log(1-positive_case_rate)
        pos_df = pd.read_csv(pos_dict, index_col=0)
        pos_df.columns = ['pos_prob']
        neg_df = pd.read_csv(neg_dict, index_col=0)
        neg_df.columns = ['neg_prob']

        self.data = pos_df.join(neg_df, how='outer')
        self.data['word_prob'] = self.data['pos_prob']+self.data['neg_prob']

        # Make prob distributions and store log of them
        for col in self.data.columns:
            self.data[col] = np.log(self.data[col]/self.data[col].sum())

        self.data = self.data.fillna(self.data['word_prob'].min())

    def classify(self, sentence):
        sentence = [x for x in sentence if x in self.data.index]
        sums = self.data.loc[sentence].aggregate('sum')
        # using log to make it faster
        pos_log_probability = self.positive_case_rate + sums['pos_prob']
        neg_log_probability = self.negative_case_rate + sums['neg_prob']
        return pos_log_probability > neg_log_probability


if __name__ == "__main__":
    train_pos_data_path = os.path.join(
        'data', 'regex_tokens_without_stop_words', 'pos_dict.csv')
    train_neg_data_path = os.path.join(
        'data', 'regex_tokens_without_stop_words', 'neg_dict.csv')
    test_data_path = os.path.join(
        'data', 'test_regex_tokens_without_stop_words', 'stemmed_data.csv')
    classifier = BayesClassifier(
        train_pos_data_path, train_neg_data_path, 0.73679818)
    dataset = Dataset(os.path.join(test_data_path))
    dataset.load_all()

    dataset.data = dataset.data[dataset.data['stars'] != 3]
    dataset.data['gold'] = dataset.data.apply(
        lambda row: row['stars'] > 3, axis=1)
    dataset.data['predicted'] = dataset.data.apply(
        lambda row: classifier.classify(row['text']), axis=1)

    gold = dataset.data['gold']
    predicted = dataset.data['predicted']
    dataset.data.to_csv('predicted.csv')
    print(f'Accuracy: {metrics.accuracy_score(gold,predicted)}')
    print(f'F-score: {metrics.accuracy_score(gold,predicted)}')
    print(f'Confusion matrix: {metrics.confusion_matrix(gold,predicted)}')
