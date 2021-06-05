import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn_pandas import DataFrameMapper
import numpy as np
import pickle
# import nltk
import random
import string
# nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

df = pd.read_csv("../../data/semi_auto_shuffled.csv")
df_test = pd.read_csv("../../data/dataset_reddit.csv")


def set_length_of_tweet(df):
    df['post_length'] = df['text'].apply(lambda post: len(post.split()))
    return df


def calculate_sentiment_score(post):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(post)['compound']
    return sentiment


def set_sentiment(df):
    df['sentiment'] = df['text'].apply(lambda post: calculate_sentiment_score(post))
    return df


# df = set_length_of_tweet(df)
# df_test = set_length_of_tweet(df_test)
# df = set_sentiment(df)
# df_test = set_sentiment(df_test)
#
letters = string.ascii_lowercase
# df['name'] = df['name'].fillna(''.join(random.choice(letters) for _ in range(7)))
# df['description'] = df['description'].fillna(''.join(random.choice(letters) for _ in range(7)))
# df_test['name'] = df_test['name'].fillna(''.join(random.choice(letters) for _ in range(7)))
# df_test['description'] = df_test['description'].fillna(''.join(random.choice(letters) for _ in range(7)))
#
# df.to_csv('../../data/dataset_training_tabular.csv')
# df_test.to_csv('../../data/dataset_test_tabular.csv')

mapper = DataFrameMapper([
    # ('description', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=8000)),
    ('text', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=8000)),
    # ('screen_name', TfidfVectorizer(ngram_range=(2, 4), analyzer='char', max_features=8000)),
    # ('name', TfidfVectorizer(ngram_range=(2, 4), analyzer='char', max_features=8000)),
    # ('post_length', None),
    # ('sentiment', None),
])

print("Fit transform")
# mapper = pickle.load(open('mapper-tweet-semi.pkl', 'rb'))
features = mapper.fit_transform(df)
# pickle.dump(mapper, open("mapper-tweet-semi.pkl", 'wb'))
print("Transform")
features_test = mapper.transform(df_test)

# # Display top-20 tf-idf words
# feature_array = unigram_vectorizer.get_feature_names()
# tfidf_sorting = unigram_vectorizer.idf_
#
# tuples = list(zip(feature_array, tfidf_sorting))
# sorted_unigrams = sorted(tuples, key=lambda x: x[1], reverse=True)
# print(dict(sorted_unigrams[:20]).keys())
#
# Split the data into train and val to get same data as for transformers
X_train, _, y_train, _ = train_test_split(features, df['label'], train_size=0.95, random_state=42,
                                          stratify=df['label'])

labels_dict = {'unrelated': 0, 'proED': 1, 'prorecovery': 2}

y_train = y_train.values.tolist()
for i in range(len(y_train)):
    y_train[i] = labels_dict[y_train[i]]

y_test = df_test['label'].values.tolist()
for i in range(len(y_test)):
    y_test[i] = labels_dict[y_test[i]]

# clf_svm = pickle.load(open('svm-bio-tweet.pkl', 'rb'))
clf_svm = svm.LinearSVC(C=1, class_weight='balanced', max_iter=1000, random_state=42)
# clf_svm = svm.LinearSVC()
#
# defining parameter range
# param_grid = {'C': [0.1, 0.5, 1],
#               'max_iter': [500, 1000, 2000],
#               'class_weight': [None, 'balanced'],
#               'random_state': [42]}
# # BEST: C=1, class_weight=balanced, max_iter=500, random_state=42
#
# grid = GridSearchCV(clf_svm, param_grid, refit=True, verbose=3)

# fitting the model for grid search
print("Training...")

# grid.fit(X_train, y_train)
# print(grid.best_estimator_)
# preds = grid.predict(features_test)
clf_svm.fit(X_train, y_train)

# pickle.dump(clf_svm, open("svm-tweet-semi.pkl", 'wb'))

preds = clf_svm.predict(features_test)

c_report = classification_report(y_test, preds, digits=3)
print(c_report)


# def f_importances(coef, names, top=-1):
#     imp = coef
#     imp, names = zip(*sorted(list(zip(imp, names))))
#
#     # Show all features
#     if top == -1:
#         top = len(names)
#
#     plt.barh(range(top), imp[::-1][0:top], align='center')
#     plt.yticks(range(top), names[::-1][0:top])
#     plt.show()


# Specify your top n features you want to visualize.
# You can also discard the abs() function
# if you are interested in negative contribution of features
# feature_names = ['description', 'text', 'screen_name', 'name', 'post_length', 'sentiment']
# feature_names = ['text']
# f_importances(clf_svm.coef_[0], feature_names)
