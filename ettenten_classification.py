#!/usr/bin/python3
# coding: utf8

import os
from pandas.parser import CParserError
import pandas as pd
import csv
import time
import pickle
import random
import math

from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

# classifiers = [MultinomialNB(), RandomForestClassifier(), LogisticRegression(), LinearSVC(),  MLPClassifier()]
names = ["MultinomialNB", "RandomForest", "LogisticRegression", "LinearSVC", "MLP"]
testdata_ettenten = 'ettenten_valitud'


def process_directory(input_dir):
    urls = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.ela')]
    chunks = []
    for url in urls:
        new_chunk = pd.read_csv(url, sep='\t', header=None, encoding='utf-8')
        new_chunk.columns = ['label', 'file', 'text']

    # Selleks, et testida, kas algfailides on mingi kala (nt it's not np.nan)
    #     for i, value in enumerate(new_chunk['text'].values):
        #     if not isinstance(value, str):
        #         print(i, url)
        chunks.append(new_chunk)
    return chunks


def prediction(train, test, clf, fod, df):

    vec = CountVectorizer(analyzer='word', lowercase=False, token_pattern='[^\s]+', min_df=df)

    targets = train['label']
    train_data = [text for text in train['text'].values]  # if isinstance(value, str)
    test_data = [text for text in test['text'].values]

    # arvutab: sõne sagedus dokumendis/sõnede arv dokumendis
    train_counts = vec.fit_transform(train_data)
    tfidf_transformer = TfidfTransformer(use_idf=True).fit(train_counts)
    train_tf = tfidf_transformer.transform(train_counts)

    ### Klassifitseerija treenimine ja test-seti peal ennustamine ###

    test_counts = vec.transform(test_data)
    test_tf = tfidf_transformer.transform(test_counts)

    clf.fit(train_tf, targets)
    predictions = clf.predict(test_tf)
    accuracy = accuracy_score(test['label'], predictions)
    print("Accuracy score of {0}: {1}".format(str(clf).split('(')[0], accuracy))
    fod.write("Accuracy score of {0}: {1}\n".format(str(clf).split('(')[0], accuracy))
    for label, file in zip(predictions, test['file']):  # test_data,
        fod.write('%r => %s\n' % (file, label))
    #     print('%r => %s\n' % (file, label))


def testi_ettenten(vec, tfidf_transformer, data, clf, fod):
    print("Collecting testset...")
    # etTenTeni peal klassifitseerimine #
    urlid = [os.path.join(data, f) for f in os.listdir(data)]
    ettenten_data = pd.DataFrame()
    for url in urlid:
        try:
            new_chunk = pd.read_csv(url, sep='\t', header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
            new_chunk.columns = ['label', 'file', 'text']
            ettenten_data = ettenten_data.append(new_chunk)

        except CParserError as detail:
            print('Viga: %s %s' % (detail, url))
    print(clf)
    print("Starting to classify")
    encoded_labels = ettenten_data['label'].map(lambda x: 1 if x == 'mittekirjak' else 0).values
    et_data = [text for text in ettenten_data['text'].values]
    ettenten_counts = vec.transform(et_data)
    X_ettenten_tfidf = tfidf_transformer.transform(ettenten_counts)
    # predictions = clf.predict(X_ettenten_tfidf) #range kategoria
    # print(predictions)

    results = clf.predict_proba(X_ettenten_tfidf) # annb tõenäosuse

    # accuracy = accuracy_score(encoded_labels, predictions)
    # precision = precision_score(encoded_labels, predictions)
    # recall = recall_score(encoded_labels, predictions)
    # f1 = f1_score(encoded_labels, predictions)

    # print("{0}: accuracy -> {1}\tprecision -> {2}\trecall -> {3}\tF1-score -> {4}\n"
    #       .format(str(clf).split('(')[0], accuracy, precision, recall, f1))
    # fod.write("{0}: accuracy -> {1}\tprecision -> {2}\trecall -> {3}\tF1-score -> {4}\n"
    #       .format(str(clf).split('(')[0], accuracy, precision, recall, f1))

    for label, predictions, file in zip(encoded_labels, results, ettenten_data['file']):
        print('{0} {1} => {2}'.format(label, file, predictions))
        fod.write('{0} => {1}'.format(file, predictions))


    # Kasuta seda alumist juppi, kui tegu on siiski terve ettenteniga ehk siis pole labeleid kusagil.
    # for file, label in zip(ettenten_data['file'], predictions):
    #     print('%r => %s' % (file, label))
    #     fod.write('%r => %s\n' % (file, label))

def load_models():
    loaded_models = []
    filenames = ['finalized_model_MultinomialNB.sav',
                 'finalized_model_RandomForest.sav',
                 'finalized_model_LogisticRegression.sav',
                 'finalized_model_LinearSVC.sav',
                 'finalized_model_MLP.sav']
    for f in filenames:
        model = pickle.load(open(f, 'rb'))
        loaded_models.append(model)
    return loaded_models


def cross_validation(data, classifiers, df):
    # RISTVALIDEERIMINE
    rows = len(data)
    with open("klassifitseerijate_tulemused_df=%s.txt" % df, "w") as fod:
        fod.write('Cut-off on min_df=%s (tunnus jäetakse kõrvale, kui selle '
                  'sagedus dokumendis on väiksem kui %s)\n' % (df, df))
        print('Cut-off: %s' % df)
        chunk_size = math.ceil(0.1 * rows)
        chunk_start, chunk_stop = 0, chunk_size
        i = 1
        while chunk_stop <= rows + chunk_size:
            train, test = pd.DataFrame\
                              (columns=['label', 'file', 'text']), pd.DataFrame(columns=['label', 'file', 'text'])
            for train_chunk in chain(data[:chunk_start], data[chunk_stop:]):
                train = train.append(train_chunk)
            for test_chunk in data[chunk_start:chunk_stop]:
                test = test.append(test_chunk)

            print("%d)\n" % i, end=' ')
            for clf in classifiers:
                prediction(train, test, clf, fod, df)
            i += 1

            chunk_start += chunk_size
            chunk_stop += chunk_size


def main():
    classifiers = load_models()
    df = 1

    traindata_ettenten = process_directory('train3_tasak')
    random.shuffle(traindata_ettenten)

    #ristvalideerimine
    # cross_validation(traindata_ettenten, classifiers, df)

    train_data = pd.DataFrame(columns=['label', 'file', 'text'])
    train_data = train_data.append(traindata_ettenten)

    print('Df=%s' % df)
    print("Training has started...")
    vec = CountVectorizer(analyzer='word', lowercase=False, token_pattern='[^\s]+', min_df=df)
    encoded_labels = train_data['label'].map(lambda x: 1 if x == 'mittekirjak' else 0).values
    train_data = [text for text in train_data['text'].values]
    train_counts = vec.fit_transform(train_data)
    tfidf_transformer = TfidfTransformer(use_idf=True).fit(train_counts)
    train_tf = tfidf_transformer.transform(train_counts)
    print("Training has ended...")


    output_file = '%s_tulemus_df=%s.csv' % (testdata_ettenten, df)
    with open(output_file, "w") as fod:
        for name, clf in zip(names, classifiers):
            clf = CalibratedClassifierCV(clf)
            clf = clf.fit(train_tf, encoded_labels)
            testi_ettenten(vec, tfidf_transformer, testdata_ettenten, clf, fod)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

"""

"""