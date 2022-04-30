#   Computational Linguistics
#   Created by Erfan Bonyadi.
#   Copyright © 1398 Erfan Bonyadi. All rights reserved.

"""
This module contains classifiers and trainings, testing and experiments on data
"""

from pathlib import Path
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import helper
from helper import Paths
from enum import Enum
from sklearn.model_selection import train_test_split
from time import time
import datetime


class ClassifierType(Enum):
    Complement_NAIVE_BAYES = 'complement_naive_bayes'
    GAUSSIAN_NAIVE_BAYES = 'gaussian_naive_bayes'
    NAIVE_BAYES = 'naive_bayes'
    NAIVE_BAYES_V2 = 'naive_bayes_v2'
    LOGISTIC_REGRESSION = 'logistic_regression'
    LOGISTIC_REGRESSION_V2 = 'logistic_regression_v2'
    LOGISTIC_REGRESSION_V3 = 'logistic_regression_v3'
    LOGISTIC_REGRESSION_V4 = 'logistic_regression_v4'
    LOGISTIC_REGRESSION_V5 = 'logistic_regression_v5'
    LOGISTIC_REGRESSION_V6 = 'logistic_regression_v6'
    LOGISTIC_REGRESSION_V7 = 'logistic_regression_v7'
    LOGISTIC_REGRESSION_V8 = 'logistic_regression_v8'
    LOGISTIC_REGRESSION_V9 = 'logistic_regression_v9'
    LOGISTIC_REGRESSION_V10 = 'logistic_regression_v10'
    LOGISTIC_REGRESSION_V11 = 'logistic_regression_v11'
    SVM_LINEAR = 'svm_linear'
    SVM_RBF = 'svm_rbf'
    SVM_POLYNOMIAL = 'svm_polynomial'


class Classifier(object):

    def __init__(self, data_path, model_path, classifier_type):
        self.clf = None
        self.data_path = data_path
        self.model_path = model_path
        self.classifier_type = classifier_type
        self.load_model()

    def load_model(self):
        """
        :return: load trained model
        """
        # check if trained model already exists
        model_file = Path(self.model_path)
        if model_file.is_file():
            self.clf = joblib.load(self.model_path)
        # no pre trained model found, we have to make one
        else:
            self.make_classifier()

    def fit_classifier(self, documents, labels):
        t0 = time()
        # checking the classifier type
        if self.classifier_type == ClassifierType.LOGISTIC_REGRESSION:
            # Logistic Regression
            # Classifier Pipeline
            # first classifier is word based
            word_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 3), norm="l1")),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            # second classifier is character based
            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 5))),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])
            # make the classifier
            vote_clf = Pipeline([['vote-clf', VotingClassifier(estimators=[("char", char_clf), ("word", word_clf)],
                                                               weights=[2, 1], voting='soft')]])
            self.clf = vote_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V2:

            word_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), norm="l1")),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = word_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V3:

            word_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 5), norm="l1")),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = word_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V4:

            word_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(3, 5), norm="l1")),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = word_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V5:

            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 5))),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = char_clf.fit(documents, labels)
        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V6:

            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 15))),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = char_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V7:

            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 10))),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = char_clf.fit(documents, labels)
        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V8:

            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 7))),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = char_clf.fit(documents, labels)
        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V9:

            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 6))),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = char_clf.fit(documents, labels)
        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V10:

            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(2, 5))),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = char_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.LOGISTIC_REGRESSION_V11:

            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(1, 5))),
                                 ('clf_logistic', LogisticRegression(C=5, multi_class="multinomial", solver='lbfgs'))])

            self.clf = char_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.NAIVE_BAYES:
            # Naive Bayes
            # Classifier Pipeline
            # first classifier is word based
            word_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 3))),
                                 ('clf_nb', MultinomialNB())])

            # second classifier is character based
            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 5))),
                                 ('clf_nb', MultinomialNB())])
            # make the classifier
            vote_clf = Pipeline([['vote-clf', VotingClassifier(estimators=[("char", char_clf), ("word", word_clf)],
                                                               weights=[2, 1], voting='soft')]])
            self.clf = vote_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.GAUSSIAN_NAIVE_BAYES:
            features_train, features_test, labels_train, labels_test =\
                train_test_split(documents, labels, test_size=0.1, random_state=10)
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
            features_train = vectorizer.fit_transform(features_train).toarray()
            features_test = vectorizer.transform(features_test).toarray()
            model = GaussianNB()

            self.clf = model.fit(features_train, labels_train)
            score_train = model.score(features_train, labels_train)
            score_test = model.score(features_test, labels_test)
            print("\nTrain set score: ", score_train)
            print("Test set score: ", score_test)

        elif self.classifier_type == ClassifierType.Complement_NAIVE_BAYES:
            features_train, features_test, labels_train, labels_test =\
                train_test_split(documents, labels, test_size=0.1, random_state=10)
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
            features_train = vectorizer.fit_transform(features_train).toarray()
            features_test = vectorizer.transform(features_test).toarray()
            model = ComplementNB()
            self.clf = model.fit(features_train, labels_train)
            score_train = model.score(features_train, labels_train)
            score_test = model.score(features_test, labels_test)
            print("\nTrain set score: ", score_train)
            print("Test set score: ", score_test)

        elif self.classifier_type == ClassifierType.NAIVE_BAYES_V2:

            word_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1))),
                                 ('clf_nb', MultinomialNB())])

            self.clf = word_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.SVM_LINEAR:
            C = 1.0  # SVM regularization parameter

            # SVC with linear kernel
            svc = SVC(kernel='linear', C=C, probability=True)

            # First Classifier => Word based
            word_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 3), norm="l1")),
                                 ('clf_svm', svc)])

            # Second Classifier => Character based
            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 5))),
                                 ('clf_svm', svc)])
            # Make the classifier
            vote_clf = Pipeline([['vote-clf', VotingClassifier(estimators=[("char", char_clf), ("word", word_clf)],
                                                               weights=[2, 1], voting='soft')]])
            self.clf = vote_clf.fit(documents, labels)

        elif self.classifier_type == ClassifierType.SVM_RBF:
            C = 1.0  # SVM regularization parameter

            # SVC with RBF kernel
            svc = SVC(kernel='rbf', gamma=0.7, C=C, probability=True)

            # First Classifier => Word based
            word_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 3), norm="l1")),
                                 ('clf_svm', svc)])

            # Second Classifier => Character based
            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 5))),
                                 ('clf_svm', svc)])
            # Make the classifier
            vote_clf = Pipeline([['vote-clf', VotingClassifier(estimators=[("char", char_clf), ("word", word_clf)],
                                                               weights=[2, 1], voting='soft')]])
            self.clf = vote_clf.fit(documents, labels)
        elif self.classifier_type == ClassifierType.SVM_POLYNOMIAL:
            C = 1.0  # SVM regularization parameter

            # SVC with polynomial (degree 3) kernel
            svc = SVC(kernel='poly', degree=3, C=C, probability=True, gamma='scale')

            # First Classifier => Word based
            word_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 3), norm="l1")),
                                 ('clf_svm', svc)])

            # Second Classifier => Character based
            char_clf = Pipeline([('vect', TfidfVectorizer(analyzer='char', ngram_range=(3, 5))),
                                 ('clf_svm', svc)])
            # Make the classifier
            vote_clf = Pipeline([['vote-clf', VotingClassifier(estimators=[("char", char_clf), ("word", word_clf)],
                                                               weights=[2, 1], voting='soft')]])
            self.clf = vote_clf.fit(documents, labels)
        else:
            helper.pprint('ERROR: classifier is not supported')
            exit()
        duration = time() - t0
        helper.pprint('Duration time', str(datetime.timedelta(seconds=duration)))

    def cross_validate(self, documents, labels):
        # cross validation
        cv = ShuffleSplit(n_splits=7, test_size=0.25, random_state=0)

        # get scores
        scores = cross_val_score(self.clf, documents, labels, cv=cv)
        print("classifier trained with mean: {}".format(scores.mean()))
        """
        precision_scoring_micro = metrics.make_scorer(metrics.precision_score, average='micro')
        precision_scoring_macro = metrics.make_scorer(metrics.precision_score, average='macro')
        micro_precision_scores = cross_val_score(self.clf, documents, labels, scoring=precision_scoring_micro, cv=cv)
        print("classifier trained with micro precision scores: {}".format(micro_precision_scores.mean()))
        macro_precision_scores = cross_val_score(self.clf, documents, labels, scoring=precision_scoring_macro, cv=cv)
        print("classifier trained with macro precision scores: {}".format(macro_precision_scores.mean()))

        recall_scoring_micro = metrics.make_scorer(metrics.recall_score, average='micro')
        recall_scoring_macro = metrics.make_scorer(metrics.recall_score, average='macro')
        micro_recall_scores = cross_val_score(self.clf, documents, labels, scoring=recall_scoring_micro, cv=cv)
        print("classifier trained with micro recall scores: {}".format(micro_recall_scores.mean()))
        macro_recall_scores = cross_val_score(self.clf, documents, labels, scoring=recall_scoring_macro, cv=cv)
        print("classifier trained with macro recall scores: {}".format(macro_recall_scores.mean()))

        f1_scoring_micro = metrics.make_scorer(metrics.f1_score, average='micro')
        f1_scoring_macro = metrics.make_scorer(metrics.f1_score, average='macro')
        micro_f1_scores = cross_val_score(self.clf, documents, labels, scoring=f1_scoring_micro, cv=cv)
        print("classifier trained with micro f1 scores: {}".format(micro_f1_scores.mean()))
        macro_f1_scores = cross_val_score(self.clf, documents, labels, scoring=f1_scoring_macro, cv=cv)
        print("classifier trained with macro f1 scores: {}".format(macro_f1_scores.mean()))
        """

    def make_classifier(self):
        # make the training dataset
        train_rows = helper.get_data_rows(self.data_path)

        # documents are reviews and labels are sentiment polarity labels
        documents = [tr[0] for tr in train_rows]
        labels = [int(tr[1]) for tr in train_rows]

        # fit the classifier
        self.fit_classifier(documents, labels)

        # cross validation
        self.cross_validate(documents, labels)
        # save it
        joblib.dump(self.clf, self.model_path)

    def classify(self, doc):
        # list of predicted probabilities for every class
        class_probability_list = self.clf.predict_proba([doc])[0]
        max_prob = max(class_probability_list)  # max probability
        predicted_class = self.clf.predict([doc])[0]  # predicted class
        return predicted_class, max_prob, class_probability_list

    def test(self, path):
        # testing the trained model to predict the data to study the false predictions
        rows = helper.get_data_rows(path)
        reviews = [rows[i][0] for i in range(len(rows))]
        labels = [rows[i][1] for i in range(len(rows))]
        false_count = 0

        for i in range(len(labels)):
            review = reviews[i]
            predicted_class = str(self.classify(review)[0])
            if str(labels[i]) != predicted_class:
                false_count += 1
                # helper.pprint(review, 'index:{} predicted:{} actual:{}'.format(i, predicted_class, labels[i]))
        helper.pprint('', 'false count:{}'.format(false_count))
        print(len(labels))


if __name__ == '__main__':

    # nb_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_NAIVE_BAYES, ClassifierType.NAIVE_BAYES)
    # nb_clf.test(Paths.CSV_REVIEW_SENTIMENT)
    # nb_clf.test(Paths.CSV_NORMAL_REVIEW_SENTIMENT)

    # normal_nb_clf = Classifier(Paths.CSV_NORMAL_REVIEW_SENTIMENT, Paths.MODEL_NORMAL_NAIVE_BAYES, ClassifierType.NAIVE_BAYES)
    # normal_nb_clf.test(Paths.CSV_REVIEW_SENTIMENT)
    # normal_nb_clf.test(Paths.CSV_NORMAL_REVIEW_SENTIMENT)

    # lr_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR, ClassifierType.LOGISTIC_REGRESSION)
    # lr_clf.test(Paths.CSV_REVIEW_SENTIMENT)
    # lr_clf.test(Paths.CSV_NORMAL_REVIEW_SENTIMENT)
    #
    # normal_lr_clf = Classifier(Paths.CSV_NORMAL_REVIEW_SENTIMENT, Paths.MODEL_NORMAL_LR, ClassifierType.LOGISTIC_REGRESSION)
    # normal_lr_clf.test(Paths.CSV_REVIEW_SENTIMENT)
    # normal_lr_clf.test(Paths.CSV_NORMAL_REVIEW_SENTIMENT)

    # g_nb_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_GNB, ClassifierType.GAUSSIAN_NAIVE_BAYES)
    # normal_g_nb_clf = Classifier(Paths.CSV_NORMAL_REVIEW_SENTIMENT, Paths.MODEL_NORMAL_GNB, ClassifierType.GAUSSIAN_NAIVE_BAYES)

    # comp_nb_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_COMP_NB, ClassifierType.Complement_NAIVE_BAYES)
    # normal_comp_nb_clf = Classifier(Paths.CSV_NORMAL_REVIEW_SENTIMENT, Paths.MODEL_NORMAL_COMP_NB, ClassifierType.Complement_NAIVE_BAYES)

    # lin_svm_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_SVM_LINEAR, ClassifierType.SVM_LINEAR)
    # normal_lin_svm_clf = Classifier(Paths.CSV_NORMAL_REVIEW_SENTIMENT, Paths.MODEL_NORMAL_SVM_LINEAR, ClassifierType.SVM_LINEAR)

    # rbf_svm_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_SVM_RBF, ClassifierType.SVM_RBF)
    # normal_rbf_svm_clf = Classifier(Paths.CSV_NORMAL_REVIEW_SENTIMENT, Paths.MODEL_NORMAL_SVM_RBF, ClassifierType.SVM_RBF)

    # poly_svm_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_SVM_POLYNOMIAL, ClassifierType.SVM_POLYNOMIAL)
    # normal_lin_svm_clf = Classifier(Paths.CSV_NORMAL_REVIEW_SENTIMENT, Paths.MODEL_NORMAL_SVM_LINEAR, ClassifierType.SVM_LINEAR)

    # nb_v2_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_NAIVE_BAYES_V2, ClassifierType.NAIVE_BAYES_V2)
    # lr_v2_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V2, ClassifierType.LOGISTIC_REGRESSION_V2)
    # normal_lr_v2_clf = Classifier(Paths.CSV_NORMAL_REVIEW_SENTIMENT, Paths.MODEL_NORMAL_LR_V2, ClassifierType.LOGISTIC_REGRESSION_V2)
    # lr_v3_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V3, ClassifierType.LOGISTIC_REGRESSION_V3)
    # lr_v4_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V4, ClassifierType.LOGISTIC_REGRESSION_V4)
    # lr_v5_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V5, ClassifierType.LOGISTIC_REGRESSION_V5)
    # lr_v6_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V6, ClassifierType.LOGISTIC_REGRESSION_V6)
    # lr_v7_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V7, ClassifierType.LOGISTIC_REGRESSION_V7)
    # lr_v8_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V8, ClassifierType.LOGISTIC_REGRESSION_V8)
    # lr_v9_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V9, ClassifierType.LOGISTIC_REGRESSION_V9)
    # lr_v10_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V10, ClassifierType.LOGISTIC_REGRESSION_V10)
    lr_v11_clf = Classifier(Paths.CSV_REVIEW_SENTIMENT, Paths.MODEL_LR_V11, ClassifierType.LOGISTIC_REGRESSION_V11)

    pass
