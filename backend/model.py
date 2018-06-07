from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import pymongo
import string
import numpy as np
from collections import Counter
from bson.json_util import dumps
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

from sklearn.model_selection import train_test_split


def balance(positive, negative):
    count_neg = len(negative.index)
    count_pos = len(positive.index)
    if (count_neg < count_pos):
        down_sample = count_neg / count_pos
        positive = positive.sample(frac=down_sample, random_state=123)
    elif (count_neg > count_pos):
        down_sample = count_pos / count_neg
        negative = negative.sample(frac=down_sample, random_state=123)

    # print(positive.info())

    return pd.concat([positive, negative])


def db_initialization():
    client = pymongo.MongoClient('localhost', 27017)
    db = client['yelpFinal']

    return db


def text_pre_process(text):
    stemmer = SnowballStemmer("english")

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [stemmer.stem(word.lower()) for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def reviews_pre_processing(df):
    df['preprocessed_text'] = df['text'].apply(text_pre_process)
    return df


def merge(first, second, on, how):
    df = pd.merge(first, second, on=on , how=how)
    return df


def model_LinearSVC(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(
        [('vct', TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')),
         ('chi', SelectKBest(chi2, k=1000)),
         ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

    model = pipeline.fit(X_train, y_train)

    vectorizer = model.named_steps['vct']
    chi = model.named_steps['chi']
    classifier = model.named_steps['clf']

    feature_names = vectorizer.get_feature_names()
    feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    print("accuracy score: " + str(model.score(X_test, y_test)))

    return model

def model_RandomForestRegressor(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(
        [('vct', TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')),
         ('chi', SelectKBest(chi2, k=1000)),
         ('clf', RandomForestRegressor(n_estimators = 100, random_state = 101))])

    model = pipeline.fit(X_train, y_train)

    vectorizer = model.named_steps['vct']
    chi = model.named_steps['chi']
    classifier = model.named_steps['clf']

    feature_names = vectorizer.get_feature_names()
    feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    print("accuracy score: " + str(model.score(X_test, y_test)))

    return model

def model_LinearRegression(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(
        [('vct', TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')),
         ('chi', SelectKBest(chi2, k=1000)),
         ('clf', LinearRegression())])

    model = pipeline.fit(X_train, y_train)

    vectorizer = model.named_steps['vct']
    chi = model.named_steps['chi']
    classifier = model.named_steps['clf']

    feature_names = vectorizer.get_feature_names()
    feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    print("accuracy score: " + str(model.score(X_test, y_test)))

    return model


def model_DecisionTree(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(
        [('vct', TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')),
         ('chi', SelectKBest(chi2, k=1000)),
         ('clf', DecisionTreeClassifier())])

    model = pipeline.fit(X_train, y_train)

    vectorizer = model.named_steps['vct']
    chi = model.named_steps['chi']
    classifier = model.named_steps['clf']

    feature_names = vectorizer.get_feature_names()
    feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    print("accuracy score: " + str(model.score(X_test, y_test)))

    return model


def model_MultinomialNB(X_train, X_test, y_train, y_test):
    pipeline = Pipeline(
        [('vct', TfidfVectorizer(min_df=.0025, max_df=.1, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')),
         ('chi', SelectKBest(chi2, k=1000)),
         ('clf', MultinomialNB())])

    model = pipeline.fit(X_train, y_train)

    vectorizer = model.named_steps['vct']
    chi = model.named_steps['chi']
    classifier = model.named_steps['clf']

    feature_names = vectorizer.get_feature_names()
    feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    print("accuracy score: " + str(model.score(X_test, y_test)))

    return model

def test_the_model(model):
    while True:
        print()
        i = input("Enter text (or Enter to quit): ")
        if i.lower() == 'exit':
            break
        result = model.predict([i])
        if result[0] == 1:
            print(" -> Positive")
        else:
            print(" -> Negative")

    return 0


def model_creation():
    db = db_initialization()

    print("Reading reviews ...")
    print()
    data_reviews = db["Italian_Reviews"].find()
    yelp_reviews = pd.DataFrame(list(data_reviews))
    #print(yelp_reviews.info())

    print("Reading restaurants ...")
    print()
    data_restaurants = db["Italian_Restaurants"].find()
    yelp_restaurants = pd.DataFrame(list(data_restaurants))
    #print(yelp_restaurants.info())

    print("Merging reviews with restaurants ...")
    print()
    merged_revies_restaurants = merge(yelp_reviews, yelp_restaurants, "business_id", "inner")
    #print(merged_revies_restaurants.info())

    print("Computing the text length of each review ...")
    print()
    merged_revies_restaurants['text_length'] = merged_revies_restaurants['text'].apply(len)
    #print(merged_revies_restaurants.info())

    print("Splitting the dataset into positive and negative based on stars ...")
    print()
    negative = merged_revies_restaurants[merged_revies_restaurants.apply(lambda x: 1 == x['stars_x'] or 2 == x['stars_x'], axis=1)]
    negative_row_indices = negative.index
    new_negative = negative.loc[negative_row_indices, :]
    new_negative['sentiment'] = 0
    # print(negative.head())
    # print()


    positive = merged_revies_restaurants[merged_revies_restaurants.apply(lambda x: 4 == x['stars_x'] or 5 == x['stars_x'], axis=1)]
    positive_row_indices = positive.index
    new_positive = positive.loc[positive_row_indices , :]
    new_positive['sentiment'] = 1
    # print(positive.head())
    # print()

    print("Balancing the dataset ...")
    print()
    balanced_dataframe = balance(new_positive, new_negative)
    # print(balanced_dataframe.head())
    # print()

    # print("Text preprocessing ...")
    # print()
    # #balanced_dataframe = reviews_pre_processing(balanced_dataframe)

    balanced_dataframe['Preprocessed_Text'] = [" ".join(review) for review in balanced_dataframe['Preprocessed_Text'].values]

    X_train, X_test, y_train, y_test = train_test_split(balanced_dataframe['Preprocessed_Text'], balanced_dataframe['sentiment'],
                                                        test_size=0.2)

    print("Training the model ...")
    model = model_LinearSVC(X_train, X_test, y_train, y_test) # Good resutls
    #model = model_RandomForestRegressor(X_train, X_test, y_train, y_test)
    #model = model_LinearRegression(X_train, X_test, y_train, y_test)
    #model = model_DecisionTree(X_train, X_test, y_train, y_test)
    #model = model_MultinomialNB(X_train, X_test, y_train, y_test) # Good results


    #test_the_model(model)
    #print(model.predict(['that was an awesome place. Great food!!!']))
    #print(model.predict(['I had a awful experience with my food. The service was mean and not polite!']))

    return model
