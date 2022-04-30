#   Computational Linguistics
#   Created by Erfan Bonyadi.
#   Copyright © 1398 Erfan Bonyadi. All rights reserved.

"""
This module contains helper functions like reading from file and paths of files to read or write.
"""
import csv
import pandas as pd


def pprint(text, title=''):
    print()
    print('*****************************************************')
    print('*******************************{}'.format(title))
    print('*****************************************************')
    print(text)


def get_data_rows(data_path):
    """
    :return:
    """
    data_rows = []

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data_rows.append(row)

    return data_rows


def write_to_csv(file_path, rows, headers=None):
    """
    :param file_path: file path to write
    :param headers: list of headers elements
    :param rows: list of rows to write, every row is a list of elements
    :return:
    """
    with open(file_path, 'w', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        if headers is not None:
            csv_writer.writerow(headers)
        for row in rows:
            csv_writer.writerow(row)


def read_csv(path):
    return pd.read_csv(path)


class Paths(object):
    CSV_TOTAL_REVIEW_WITH_SUGGESTIONS = 'data/totalReviewWithSuggestion.csv'
    CSV_REVIEW_SENTIMENT = 'data/review_sentiment.csv'
    CSV_VOCAB = 'data/vocab.csv'
    CSV_NORMAL_VOCAB = 'data/normal_vocab.csv'
    CSV_NORMAL_REVIEW_SENTIMENT = 'data/normal_review_sentiment.csv'

    MODEL_NAIVE_BAYES = 'models/naive_bayes_model.pkl'

    # trained model after normalize, removing stopwords, 1500 first frequent words in corpus
    MODEL_NORMAL_NAIVE_BAYES = 'models/normal_naive_bayes_model.pkl'
    MODEL_GNB = 'models/gnb_model.pkl'  # gaussian naive bayes
    MODEL_NORMAL_GNB = 'models/normal_gnb_model.pkl'  # gaussian naive bayes with normal data

    MODEL_COMP_NB = 'models/complement_nb_model.pkl'  # Complement naive bayes
    MODEL_NORMAL_COMP_NB = 'models/normal_complement_nb_model.pkl'  # Complement naive bayes with normal data

    MODEL_LR = 'models/lr_model.pkl'
    MODEL_NORMAL_LR = 'models/normal_lr_model.pkl'

    MODEL_SVM_LINEAR = 'models/model_svm_linear.pkl'
    MODEL_SVM_RBF = 'models/model_svm_rbf.pkl'
    MODEL_SVM_POLYNOMIAL = 'models/model_svm_polynomial.pkl'

    MODEL_NORMAL_SVM_LINEAR = 'models/model_normal_svm_linear.pkl'
    MODEL_NORMAL_SVM_RBF = 'models/model_normal_svm_rbf.pkl'
    MODEL_NORMAL_SVM_POLYNOMIAL = 'models/model_normal_svm_polynomial.pkl'

    MODEL_NAIVE_BAYES_V2 = 'models/model_naive_bayes_v2.pkl'
    MODEL_LR_V2 = 'models/model_lr_v2.pkl'
    MODEL_NORMAL_LR_V2 = 'models/model_normal_lr_v2.pkl'
    MODEL_LR_V3 = 'models/model_normal_lr_v3.pkl'
    MODEL_LR_V4 = 'models/model_normal_lr_v4.pkl'
    MODEL_LR_V5 = 'models/model_normal_lr_v5.pkl'
    MODEL_LR_V6 = 'models/model_normal_lr_v6.pkl'
    MODEL_LR_V7 = 'models/model_normal_lr_v7.pkl'
    MODEL_LR_V8 = 'models/model_normal_lr_v8.pkl'
    MODEL_LR_V9 = 'models/model_normal_lr_v9.pkl'
    MODEL_LR_V10 = 'models/model_normal_lr_v10.pkl'
    MODEL_LR_V11 = 'models/model_normal_lr_v11.pkl'
