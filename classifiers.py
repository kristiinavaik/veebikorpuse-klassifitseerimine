#!/usr/bin/python3
# coding: utf8

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib



def process_directory(input_dir):
    urls = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.ela')]
    chunks = []
    for url in urls:
        new_chunk = pd.read_csv(url, sep='\t', header=None, encoding='utf-8')
        new_chunk.columns = ['label', 'file', 'text']
        chunks.append(new_chunk)
    return chunks

def test(train, df): # clf
    vec = CountVectorizer(analyzer='word', lowercase=False, token_pattern='[^\s]+', min_df=df)
    targets = train['label']
    train_data = [text for text in train['text'].values]

    # teeb tf-idf normaliseerimist
    train_counts = vec.fit_transform(train_data)
    # vocab = vec.get_feature_names() # annab vektori pikkuse
    tfidf_transformer = TfidfTransformer(use_idf=True).fit(train_counts)
    train_tf = tfidf_transformer.transform(train_counts)

    # clf = clf.fit(train_tf, targets)

def most_informative_feature_for_SVC(vectorizer, classifier, n):
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()

    # SVC jaoks
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]
    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)
    print('----------------------------------')
    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


def most_informative_feature_for_MLP(vectorizer, classifier, classlabel, n):
    labelid = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names()
    topn = sorted(zip(classifier.coefs_[labelid], feature_names))[:n]
    for coef, feat in topn:
        print(classlabel, feat, coef)

def main():
    data = process_directory('train3_tasak')
    train_data = pd.DataFrame(columns=['label', 'file', 'text'])
    train_data = train_data.append(data)

    clf = LinearSVC()
    # clf = MLPClassifier()
    # clfs = [LinearSVC(), MLPClassifier()]

    df = 0

    while df < 11:
        print('Df=%s' % df)
        targets = train_data['label'].values
        vec = CountVectorizer(analyzer='word', lowercase=False, token_pattern='[^\s]+', min_df=df)

        train_values = [text for text in train_data['text'].values]
        train_counts = vec.fit_transform(train_values)
        tfidf_transformer = TfidfTransformer(use_idf=True).fit(train_counts)
        train_tf = tfidf_transformer.transform(train_counts)

        clf.fit(train_tf, targets)
        most_informative_feature_for_SVC(vec, clf, 50)

        # joblib.dump(clf, 'saved_mlp_clf.pkl', compress=3)
        # clf = joblib.load('saved_mlp_clf.pkl')
        # most_informative_feature_for_MLP(vec, clf, 'uusmeedia', 20)
        # most_informative_feature_for_MLP(vec, clf, 'kirjak', 20)
        df += 1




if __name__ == '__main__':
    main()
