import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


real_train = ['op_spam_v1.4/negative_polarity/truthful_from_Web/fold1',
              'op_spam_v1.4/negative_polarity/truthful_from_Web/fold2',
              'op_spam_v1.4/negative_polarity/truthful_from_Web/fold3',
              'op_spam_v1.4/negative_polarity/truthful_from_Web/fold4']

fake_train = ['op_spam_v1.4/negative_polarity/deceptive_from_MTurk/fold1',
              'op_spam_v1.4/negative_polarity/deceptive_from_MTurk/fold2',
              'op_spam_v1.4/negative_polarity/deceptive_from_MTurk/fold3',
              'op_spam_v1.4/negative_polarity/deceptive_from_MTurk/fold4']


real_test = ['op_spam_v1.4/negative_polarity/truthful_from_Web/fold5']

fake_test = ['op_spam_v1.4/negative_polarity/deceptive_from_MTurk/fold5']


def read_data(folds, label):

    reviews = []
    labels = []

    for i in folds:
        for j in os.listdir(i):
            with open(os.path.join(i, j), 'r', encoding='utf-8') as file:
                reviews.append(file.read())
                labels.append(label)
                
    df = pd.DataFrame({'review': reviews, 'label': labels})
    
    return df


real_train_data = read_data(real_train, 1) 
real_test_data = read_data(real_test, 1)
fake_train_data = read_data(fake_train, 0)
fake_test_data = read_data(fake_test, 0)


train_data = pd.concat([real_train_data, fake_train_data], axis=0)
test_data = pd.concat([real_test_data, fake_test_data], axis=0)

vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_data['review'])
y_train = train_data['label']
X_test = vectorizer.transform(test_data['review'])
y_test = test_data['label']