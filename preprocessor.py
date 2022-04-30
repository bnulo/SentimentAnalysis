#   Computational Linguistics
#   Created by Erfan Bonyadi.
#   Copyright © 1398 Erfan Bonyadi. All rights reserved.

"""
This module contains functions used to preprocess.
"""

from collections import Counter
import re
import hazm
import helper
from helper import Paths


def correct_encoding(text):
    # تصحیح انکدینگ
    text = text.replace("ة", "ه")
    text = text.replace('ك', 'ک')
    text = text.replace('ي', 'ی')
    return text


def remove_stopwords(text_list, vocab_list=None):
    return_list = []

    stopwords_list = ['ی', 'ک', 'ام', 'برا', 'ب', 'ای', 'بی', 'آر', 'اس', 'زد', 'دی', 'جی', 'ال', 'ش', 'تی', 'اچ', 'اب', 'ان', 'ایی', 'نا', 'آی', 'اف', 'ایه', 'پا', 'م', 'ها', 'های']
    stopwords_list.extend(hazm.stopwords_list())

    if vocab_list is None:
        for text in text_list:
            if text == '':
                continue
            if text not in stopwords_list:
                return_list.append(text)
    else:
        for text in text_list:
            if text == '':
                continue
            if text not in stopwords_list and text in vocab_list:
                return_list.append(text)
    return return_list


def normalize(text):
    text = correct_encoding(text)
    text = re.sub(r'[a-zA-Z0-9۰-۹]', ' ', text)
    text = text.replace('\u200c', '')
    text = text.replace('\ufeff', '')
    text = text.replace('\u200e', '')
    text = text.replace('\u200f', '')
    text = text.replace('\u200d', '')
    text = text.replace('دی جی کالا', 'دیجیکالا')
    text = text.replace('ال سی دی', 'السیدی')
    text = text.replace('العادست', 'العاده')
    text = text.replace('العادس', 'العاده')
    text = text.replace('العادن', 'العاده')
    text = text.replace('فوق العاده', 'فوقالعاده')

    text = text.replace('نرم افزار', 'نرمافزار')
    text = text.replace('سخت افزار', 'سختافزار')

    text = text.replace('دیجی کالا', 'دیجیکالا')
    text = text.replace('دیجی ', 'دیجیکالا')
    reg = re.compile(r"(دیجیکالا)(\S)")  # بعد از کلمه‌ی دیجیکالا یک فاصله گذاشته شد
    text = re.sub(reg, r"\1 \2", text)
    reg = re.compile(r"(\S)(دیجیکالا)")  # قبل از کلمه‌ی دیجیکالا یک فاصله گذاشته شد
    text = re.sub(reg, r"\1 \2", text)

    text = text.replace('', '')
    text = text.replace('لپتاب', 'لپتاپ')
    text = text.replace('لبتاب', 'لپتاپ')
    text = text.replace('وای فای', 'وایفای')
    text = text.replace('سیستم عامل', 'سیستمعامل')
    text = text.replace('شگفت انگیز', 'شگفتانگیز')
    text = text.replace('سی دی', 'سیدی')
    text = text.replace('دی وی دی', 'دیویدی')
    text = text.replace('اچ دی', 'اچدی')
    text = text.replace('اس اس دی', 'اساسدی')

    reg = re.compile(r"می (\S+\s)")  # فاصله‌ی بعد از می حذف شد
    text = re.sub(reg, r"می\1", text)
    reg = re.compile(r"\s(های?\s)")  # فاصله‌ی قبل از ها و های حذف شد
    text = re.sub(reg, r"\1", text)
    reg = re.compile(r"\s(تر\s)")  # فاصله‌ی قبل از تر حذف شد
    text = re.sub(reg, r"\1", text)
    reg = re.compile(r"\s(ترین\s)")  # فاصله‌ی قبل از ترین حذف شد
    text = re.sub(reg, r"\1", text)

    return text


def normalize_remove_stopwords(text):
    text = normalize(text)
    tokens = text.split(' ')
    word_freq_list = helper.get_data_rows(Paths.CSV_NORMAL_VOCAB)[:1500]
    vocab_list = [word_freq_list[i][0] for i in range(len(word_freq_list))]
    normal_list = remove_stopwords(tokens, vocab_list)
    text = ' '.join(normal_list)
    return text


class Preprocessor(object):

    @staticmethod
    def study_the_data():
        """
        in this method we study the data and relation between suggestions and scores
        and for conclusion: we suppose suggestions as sentiment labels
        and remove scores column
        :return:
        """
        """ 

        data = pd.read_csv(Paths.PATH_TOTAL_REVIEW_WITH_SUGGESTIONS)
        scores = data['Score']
        suggestions = data['Suggestion']
        # scores_1 = []
        # for i in range(len(scores)):
        #     if suggestions[i] == 1 and scores[i] < 50:
        #         scores_1.append(i)
        # pprint(scores_1)
        # pprint(len(scores_1), 'len')
        # pprint(max(scores_1), 'max')
        # pprint(min(scores_1), 'min')
        # pprint(np.average(scores_1), 'mean')

        #################
        پیشنهاد خرید داده‌اند
        score == 1: POSITIVE
        count: 2382
        max: 100
        min: 0
        mean: 83
        #################
        پیشنهاد دادند خریداری نشود
        score == 2: NEGATIVE
        count: 419
        max: 100
        min: 0
        mean: 63
        #################
        نظرشان برای خرید خنثی بوده است
        score == 3: NEUTRAL
        count: 460
        max: 100
        min: 0
        mean: 45
        """
        pass

    @staticmethod
    def create_sentiment_csv():
        """
        make new .csv file with 2 columns:
        1: reviews
        2: sentiment labels: 1: positive, 2: negative, 3: neutral
        :return:
        """
        data = helper.read_csv(Paths.CSV_TOTAL_REVIEW_WITH_SUGGESTIONS)
        reviews = data['Text']
        labels = data['Suggestion']
        helper.write_to_csv(Paths.CSV_REVIEW_SENTIMENT, [[reviews[i], labels[i]] for i in range(len(reviews))])

    @staticmethod
    def make_all_words_dict_csv():
        reviews = helper.read_csv(Paths.CSV_TOTAL_REVIEW_WITH_SUGGESTIONS)['Text']
        all_text = ' '.join(reviews)
        all_text = normalize(all_text)
        words = all_text.split(' ')

        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        vocab = remove_stopwords(vocab)
        helper.write_to_csv(Paths.CSV_NORMAL_VOCAB, [[vocab[i],
                                                      counts[vocab[i]]] for i in range(len(vocab))])

    @staticmethod
    def make_stopwords_csv():
        file = open('data/STOPWORDS', "r", encoding="utf-8")
        return_string = file.read()
        file.close()
        words = return_string.strip().split('\n')
        helper.write_to_csv('data/stopwords.csv', [[words[i]] for i in range(len(words))])

    @staticmethod
    def make_normal_review_sentiment_csv():
        review_sentiments_rows = helper.get_data_rows(Paths.CSV_REVIEW_SENTIMENT)
        normal_rows = [[normalize_remove_stopwords(review_sentiments_rows[i][0]),
                        review_sentiments_rows[i][1]] for i in range(len(review_sentiments_rows))]
        helper.write_to_csv(Paths.CSV_NORMAL_REVIEW_SENTIMENT, normal_rows)


if __name__ == '__main__':
    # Preprocessor.make_all_words_dict_csv()
    # Preprocessor.make_stopwords_csv()
    # Preprocessor.make_normal_review_sentiment_csv()
    pass
