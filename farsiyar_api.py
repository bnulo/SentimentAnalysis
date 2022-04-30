#   Computational Linguistics
#   Created by Erfan Bonyadi.
#   Copyright © 1398 Erfan Bonyadi. All rights reserved.

"""
This module is used for
request every comment to https://app.text-mining.ir/ API,
get the sentiment
and double check the sentiment with first corpus
"""
import helper
from helper import Paths
import requests
import json

API_KEY = ""
APP_NAME = ""
BASE_URL = "http://api.text-mining.ir/api/"
SENTIMENT_URL = BASE_URL + "SentimentAnalyzer/SentimentClassifier"

# read corpus
train_rows = helper.get_data_rows(Paths.CSV_REVIEW_SENTIMENT)
# documents are reviews and labels are sentiment polarity labels
documents = [tr[0] for tr in train_rows]
old_labels = [int(tr[1]) for tr in train_rows]

# RESPONSE:   0:negative 1:neutral 2:positive
# OUR LABELS: 1:positive 2:negative 3:neutral


def call_api(url, data, token):
    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer " + token,
        'Cache-Control': "no-cache"
    }
    response = requests.request("POST", url, data=data.encode("utf-8"), headers=headers)
    helper.pprint(data)
    helper.pprint(response)
    helper.pprint(response.text)
    helper.pprint(response.content)
    return response.text


def get_token(api_key):
    """
    Get Token by Api Key
    param: api_key
    return: token
    """
    url = BASE_URL + "Token/GetToken"
    querystring = {"apikey": api_key}
    response = requests.request("GET", url, params=querystring)
    data = json.loads(response.text)
    token = data['token']
    helper.pprint(token)
    return token


payload = u"\"اصلا خوب نبود\""
payload = '"افتضاح"'
if __name__ == '__main__':
    token_key = get_token(API_KEY)
    # call_api(SENTIMENT_URL, payload, token_key)
    url = BASE_URL + "LanguageDetection/Predict"
    payload = u'"شام ییبسن یا یوخ. سن سیز بوغازیمنان گتمیر شام. به به نه قشه یردی. ساغ اول سیز نئجه سیز. نئجه سن؟ اوشاقلار نئجه دیر؟ سلام لاری وار سیزین کی لر نئجه دیر. یاخچی"'
    payload = u'"سلام برای امتحان زبان این متن را ارسال میکنم"'
    print(call_api(url, payload, token_key))
    pass
