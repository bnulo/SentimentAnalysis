# Sentiment analysis on digikala data 


## Phase 1

The data and relation between suggestions about buying the goods or not, and scores studied and for conclusion:
suggestions supposed to be as sentiment labels and scores column removed.

Resource of Corpus: https://github.com/minasmz/Sentiment-Analysis-with-LSTM-in-Persian
#### Sugggested to buy
~~~
score == 1: POSITIVE
count: 2382
max: 100
min: 0
mean: 83
~~~
#### Sugggested NOT to buy
~~~
score == 2: NEGATIVE
count: 419
max: 100
min: 0
mean: 63
~~~
#### not Suggested to buy nor not to buy
~~~
score == 3: NEUTRAL
count: 460
max: 100
min: 0
mean: 45
~~~

## Phase 2: Before Normalization
- review_sentiment.csv made from totalReviewWithSuggestion.csv
- 12289 tokens were in the corpus before any normalization.
- classifier trained with cross validation: ```mean_score=0.7389705882352942, n_splits=7, test_size=0.25```
- stopwords.csv file made from STOPWORDS
- naive_bayes_model.pkl made by classifier
## Phase 3
- vocabulary made from corpus in vocab.csv file
- normal_review_sentiment.csv made from review_sentiment.csv after normalizing and removing
some stop words. normal_review_sentiment.csv includes first 1500 most frequent words in the corpus
- classifier trained with cross validation: mean_score=0.7389705882352942, n_splits=7, test_size=0.25
- naive_bayes_model.pkl made by classifier
## Scores

### classifier trained with these mean cross validation scores:
#### 1. Mean Score:
    - 0.7389705882352942
#### 2. Mean Precision Scores
    - micro: 0.7389705882352942
    - macro: 0.24632352941176472   

#### 3. Mean Recall Scores:
    - micro recall scores: 0.7389705882352942
    - macro recall scores: 0.3333333333333333  

#### 4. Mean F1 Scores:
    - micro: 0.7389705882352943
    - macro: 0.2832694901118499



#### Scores for every single round

| Score \ Round  | 1          | 2          | 3         | 4          | 5          | 6          |
|----------------|------------|------------|-----------|------------|------------|------------|
| 0.75367647     | 0.70710784 | 0.74632353 | 0.7377451 | 0.73897059 | 0.75490196 | 0.73406863 |

#### NOTE
> Changing the amount of the data, changed scores, but
normalization and removing stopwords kept them unchanged.

## Phase 4
### Checking and Correcting the Sentiment Labels


1. About 25 wrong predictions were studied. Some labels were corrected and changed.
2. logestic regression classifier trained
    * ```mean: 0.7762605042016805```
    * ```mean: 0.7764355742296919 for normalized data```

3. Improvements were seen in naive bayes classifier after label correction
    * ```mean: 0.7400210084033614```
        * 875 false predictions
    * ```mean: 0.7400210084033614 after normalize```

| data / classifier  | Logistic Regression  | Naive Bayes       |
|--------------------|----------------------|-------------------|
| Dat                | 0.7762605042016805   | 7400210084033614  |
| Normalized Data    | 0.7764355742296919   | 7400210084033614  |

## Phase 5
1. logistic regression were tested on data to get some false predictions
2. 161 false predictions studied and their labels revised and in majority of  cases labeling was wrong, so corrected.
3. classifiers trained with data and normalized data with new labels
### Scores

| data / classifier | Logistic Regression | Naive Bayes        |
|-------------------|---------------------|--------------------|
| Dat               | 0.8040966386554621  | 0.769782913165266  |
| Normalized Data   | 0.8063725490196079  | 0.769782913165266  |  


 ### False Predictions Count, out of 3261 documents
| Classifier / Test on                             | Data  | Normalized Data |  
|--------------------------------------------------|-------|-----------------|
| Logistic Regression trained with data            | 109   | 576             |
| Logistic Regression trained with normalized data | 560   | 213             |
| Naive Bayes trained with data                    | 776   | 776             |
| Naive Bayes trained with normalized data         | 776   | 776             |

## Gaussian Naive Bayes Scores ``test_size=0.1, random_state=10``
~~~

Train set score:  0.8248125426039536
Test set score:  0.6330275229357798
~~~
## Gaussian Naive Bayes Scores ``normalized data test_size=0.1, random_state=10``
~~~
Train set score:  0.5170415814587593
Test set score:  0.4036697247706422
~~~

## Complement Naive Bayes Scores ``test_size=0.1, random_state=10``
~~~
Train set score:  0.8057259713701431
Test set score:  0.7767584097859327
~~~
## Complement Naive Bayes Scores ``normalized data test_size=0.1, random_state=10``
~~~
Train set score:  0.8265167007498296
Test set score:  0.7033639143730887
~~~

## SVM Linear kernel
~~~
classifier trained with
time: 0:02:45.553099
mean: 0.8221288515406163
normalized data and mean: 0.8091736694677872
~~~

## SVM RBF kernel
~~~
time: 0:03:44.719173
mean Score: 0.8296568627450981
~~~

## SVM Polynomial kernel
~~~
time: 0:04:07.382019
mean Score: 0.7949929971988795
~~~

* setting max_df=0.5 for initializing Naive Bayes and Logistic Regression
* kept Naive Bayes Score unchanged and Logistic Regression Score decreased (0.00017507002801120386).
* In Naive Bayes V2, 1gram word based tfidf vectorizer used.
* In Logistic Regression V2, 1gram word based tfidf vectorizer used.
* In Logistic Regression V3, (1 -> 5)-gram word based tfidf vectorizer used
* In Logistic Regression V4, (3 -> 5)-gram word based tfidf vectorizer used
* In Logistic Regression V5, (3 -> 5)-gram character based tfidf vectorizer used
* In Logistic Regression V6, (3 -> 15)-gram character based tfidf vectorizer used
* In Logistic Regression V7, (3 -> 10)-gram character based tfidf vectorizer used
* In Logistic Regression V8, (3 -> 7)-gram character based tfidf vectorizer used
* In Logistic Regression V9, (3 -> 6)-gram character based tfidf vectorizer used
* In Logistic Regression V10, (2 -> 5)-gram character based tfidf vectorizer used
* In Logistic Regression V11, (1 -> 5)-gram character based tfidf vectorizer used

|                                               | Time           | Mean Score           |
|-----------------------------------------------|----------------|----------------------|
 | Naive Bayes                                   | 0:00:03.901275 | 0.769782913165266    |
 | Naive Bayes V2                                | 0:00:00.165345 | 0.769782913165266    |
 | Logistic Regression                           | 0:00:21.461250 | 0.8039215686274509   |
 | Logistic Regression V2                        | 0:00:00.862117 | 0.7729341736694677   |
 | Logistic Regression V2 (on Normalized data)   | 0:00:00.278024 | 0.7823879551820728   |
 | Logistic Regression V3                        | 0:00:15.747416 | 0.769782913165266    |
 | Logistic Regression V4                        | 0:00:13.402116 | 0.7710084033613445   |
 | Logistic Regression V5                        | 0:00:15.329675 | 0.8249299719887955 * |
 | Logistic Regression V6                        | 0:04:38.927389 | 0.8044467787114845   |
 | Logistic Regression V7                        | 0:02:03.405992 | 0.810049019607843    |
 | Logistic Regression V8                        | 0:00:45.646399 | 0.8182773109243698   |
 | Logistic Regression V9                        | 0:00:24.831395 | 0.8219537815126049   |
 | Logistic Regression V10                       | 0:00:16.128916 | 0.8247549019607842   |
 | Logistic Regression V11                       | 0:00:17.159857 | 0.8224789915966386   |
 | SVM Linear kernel                             | 0:02:45.553099 | 0.8221288515406163   |
 | SVM RBF kernel                                | 0:03:44.719173 | 0.8296568627450981   |
 | SVM Polynomial kernel                         | 0:04:07.382019 | 0.7949929971988795   |

|               | Time           | Train Mean Score   | Test Mean Score    |
|---------------|----------------|--------------------|--------------------|
| Gaussian NB   | 0:00:02.659921 | 0.8248125426039536 | 0.6330275229357798 |
| Complement NB | 0:00:00.469990 | 0.8057259713701431 | 0.7767584097859327 |

## Conclusion
With the same data sets in this classification task with our vectorization method
* Normalizing and removing stopwords did not change the scores meaningfully
* SVM training duration time is significantly longer than Naive Bayes and Logistic Regression
* Using n-grams for vectorizing is beneficent for this task
* In the Naive Bayes, the amount of data has the most impact on the scores.
