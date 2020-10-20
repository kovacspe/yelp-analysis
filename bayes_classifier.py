import pandas as pd
import numpy as np


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


classifier = BayesClassifier(
    'regex_tokens\\pos_dict.csv', 'regex_tokens\\neg_dict.csv', 0.5)
