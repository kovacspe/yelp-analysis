from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
import json
import pandas as pd
import numpy as np
import os
from dataset import read_json, DATA_FILES

# nltk.download('wordnet')
# nltk.download('punkt')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
print('the' in stop_words)
tokenizer = RegexpTokenizer(r'\w+')


def tokenize(sentence, drop_stop_words=False):
    tokens = tokenizer.tokenize(sentence)
    if drop_stop_words:
        return [x for x in tokens if not x in stop_words]
    else:
        return tokens


def lemmatize(sentence):
    return [lemmatizer.lemmatize(word) for word in sentence]


def stemmer(sentence):
    return [porter.stem(word) for word in sentence]


class FreqDict(dict):
    def insert_word(self, word):
        if word in self:
            self[word] += 1
        else:
            self[word] = 1

    def to_sorted_pandas_df(self):
        df = pd.DataFrame.from_dict(self, orient='index')
        df.columns = ['count']
        df.sort_values(by=['count'], ascending=False, inplace=True)
        return df


class SentimentDict(dict):
    def insert_word(self, word, stars):
        if word in self:
            self[word].append(stars)
        else:
            self[word] = [stars]

    def aggregate_pandas_df(self):
        mean_std_dict = {}
        for key, value in self.items():
            v = np.array(value)
            # filter rare occurences
            if len(v) > 20:
                mean_std_dict[key] = [np.mean(v), np.std(v), len(v)]
        df = pd.DataFrame.from_dict(mean_std_dict, orient='index')
        df.columns = ['mean', 'std', 'count']
        df.sort_values(by=['std'], ascending=True, inplace=True)
        return df


def preprocess_review_dataset(file, output_name, num_reviews, skip_first=0, drop_stop_word=False):
    word_dict = FreqDict()
    pos_dict = FreqDict()
    neg_dict = FreqDict()
    useful_dict = FreqDict()
    useless_dict = FreqDict()
    sentiment = SentimentDict()
    usefullness = SentimentDict()

    if not os.path.exists(output_name):
        os.mkdir(output_name)
    stemmed_data_path = os.path.join(output_name, 'stemmed_data.csv')
    # First read to get stem words and get word dictionary
    with open(file, 'r', encoding='utf-8') as f:
        with open(stemmed_data_path, 'w', encoding='utf-8') as out:
            for i, line in enumerate(f):
                if i < skip_first:
                    continue
                review = json.loads(line)
                text = stemmer(tokenize(review['text'], drop_stop_word))
                for word in text:
                    word_dict.insert_word(word)
                    sentiment.insert_word(word, review['stars'])
                    usefullness.insert_word(word, review['useful'])
                    if review['stars'] > 3:
                        pos_dict.insert_word(word)
                    elif review['stars'] < 3:
                        neg_dict.insert_word(word)
                    if review['useful'] >= 1:
                        useful_dict.insert_word(word)
                    elif review['useful'] < 1:
                        useless_dict.insert_word(word)
                print(json.dumps({
                    'text': text,
                    'stars': review['stars'],
                    'useful': review['useful']
                }), file=out)
                if i+1 >= skip_first+num_reviews:
                    break
                if i % 5000 == 0:
                    print(
                        f'Processed reviews: {100*(i-skip_first)/num_reviews:2f}%')

    # Cutoff low freqency words
    df = word_dict.to_sorted_pandas_df()
    sentiment.aggregate_pandas_df()
    # 1 - unknown word
    # 0 - padding
    df['id'] = np.arange(len(df)) + 2
    df = df[df['count'] > 10]
    word_dict = df['id'].to_dict()

    # Save dictionaries
    pos_dict.to_sorted_pandas_df().to_csv(os.path.join(output_name, 'pos_dict.csv'))
    neg_dict.to_sorted_pandas_df().to_csv(os.path.join(output_name, 'neg_dict.csv'))
    useful_dict.to_sorted_pandas_df().to_csv(
        os.path.join(output_name, 'useful_dict.csv'))
    useless_dict.to_sorted_pandas_df().to_csv(
        os.path.join(output_name, 'useless_dict.csv'))
    df.to_csv(os.path.join(output_name, 'word_dict.csv'))
    sentiment.aggregate_pandas_df().to_csv(
        os.path.join(output_name, 'sentiment.csv'))
    usefullness.aggregate_pandas_df().to_csv(
        os.path.join(output_name, 'usefullness.csv'))

    # Inicialize np.arrays
    inp = np.zeros((num_reviews), dtype=object)
    stars = np.array(np.zeros(num_reviews))
    useful = np.array(np.zeros(num_reviews))
    with open(stemmed_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            cont = json.loads(line)
            text2num = np.array([word_dict.get(word, 1)
                                 for word in cont['text']])
            inp[i] = text2num
            stars[i] = cont['stars']
            useful[i] = cont['useful']
            if i % 5000 == 0:
                print(
                    f'Processed reviews: {100*i/num_reviews:2f}%')
    with open(os.path.join(output_name, 'data.npy'), 'wb') as f:
        np.save(f, inp)
        np.save(f, stars)
        np.save(f, useful)


if __name__ == "__main__":
    preprocess_review_dataset(
        DATA_FILES['review'], 'regex_tokens', 1000000)
    preprocess_review_dataset(
        DATA_FILES['review'], 'test_regex_tokens', 50000, skip_first=1000000)

    preprocess_review_dataset(
        DATA_FILES['review'], 'regex_tokens_without_stop_words', 1000000, drop_stop_word=True)
    preprocess_review_dataset(
        DATA_FILES['review'], 'test_regex_tokens_without_stop_words', 50000, skip_first=1000000, drop_stop_word=True)
