from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import json
import pandas as pd
import numpy as np
from dataset import read_json, DATA_FILES

# nltk.download('wordnet')
# nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence,)


def lemmatize(sentence):
    return [lemmatizer.lemmatize(word) for word in sentence]


def stemmer(sentence):
    return [porter.stem(word) for word in sentence]


def preprocess_review_dataset(file, output_name, num_reviews, skip_first=0):
    word_dict = {}
    # First read to get stem words and get word dictionary
    with open(file, 'r', encoding='utf-8') as f:
        with open('tmp', 'w', encoding='utf-8') as out:
            for i, line in enumerate(f):
                if i < skip_first:
                    continue
                review = json.loads(line)
                text = stemmer(tokenize(review['text']))
                for word in text:
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
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

    df = pd.DataFrame.from_dict(word_dict, orient='index')
    df.columns = ['count']
    df.sort_values(by=['count'], ascending=False, inplace=True)
    # 1 - unknown word
    # 0 - padding
    df['id'] = np.arange(len(df)) + 2
    df = df[df['count'] > 10]
    word_dict = df['id'].to_dict()

    inp = np.zeros((num_reviews), dtype=object)
    print(inp)
    stars = np.array(np.zeros(num_reviews))
    useful = np.array(np.zeros(num_reviews))
    with open('tmp', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            cont = json.loads(line)
            text2num = np.array([word_dict.get(word, 1)
                                 for word in cont['text']])
            inp[i] = text2num
            stars[i] = cont['stars']
            useful[i] = cont['useful']
            if i % 5000 == 0:
                print(
                    f'Processed reviews: {100*(i-skip_first)/num_reviews:2f}%')
    with open(f'{output_name}.npy', 'wb') as f:
        np.save(f, inp)
        np.save(f, stars)
        np.save(f, useful)


preprocess_review_dataset(DATA_FILES['review'], 'test', 100000, 1000000)
