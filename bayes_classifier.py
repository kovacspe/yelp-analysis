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
    # dataset paths
    data_path = 'data'
    train_data = os.path.join('data', 'regex_tokens')
    test_data = os.path.join('data', 'test_regex_tokens')
    train_data_stop_words = os.path.join(
        'data', 'regex_tokens_without_stop_words')
    test_data_stop_words = os.path.join(
        'data', 'test_regex_tokens_without_stop_words')
    for train, test in [(train_data, test_data), (train_data_stop_words, test_data_stop_words)]:
        # stars
        train_pos_data_path = os.path.join(train, 'pos_dict.csv')
        train_neg_data_path = os.path.join(train, 'neg_dict.csv')

        # useful
        train_useful_data_path = os.path.join(train, 'useful_dict.csv')
        train_useless_data_path = os.path.join(train, 'useless_dict.csv')

        # test set
        test_data_path = os.path.join(test, 'stemmed_data.csv')

        stars_classifier = BayesClassifier(
            train_pos_data_path, train_neg_data_path, 0.73679818)
        useful_classifier = BayesClassifier(
            train_useful_data_path, train_useless_data_path, 0.45751942)
        test_dataset = Dataset(os.path.join(test_data_path))
        test_dataset.load_all()

        types = [
            ('stars', 4, stars_classifier),
            ('useful', 1, useful_classifier)
        ]

        for t, threshold, classifier in types:
            if t == 'stars':
                test_dataset.data = test_dataset.data[test_dataset.data[t] != 3]
            # Classify
            test_dataset.data[f'{t}_gold'] = test_dataset.data.apply(
                lambda row: row[t] >= threshold, axis=1)
            test_dataset.data[f'{t}_predicted'] = test_dataset.data.apply(
                lambda row: classifier.classify(row['text']), axis=1)

            # Evaluate and print results
            gold = test_dataset.data[f'{t}_gold']
            predicted = test_dataset.data[f'{t}_predicted']
            print('--------------------------------')
            print(f'Predictor: {t}, Dataset:{test}')
            print(f'Accuracy: {metrics.accuracy_score(gold,predicted)}')
            print(
                f'Precision: {metrics.average_precision_score(gold,predicted)}')
            print(f'Recall: {metrics.recall_score(gold,predicted)}')
            print(f'F-score: {metrics.f1_score(gold,predicted)}')
            print(
                f'Confusion matrix: {metrics.confusion_matrix(gold,predicted)}')
            print('--------------------------------')
            # save test
        test_dataset.data.to_csv('predicted.csv')
