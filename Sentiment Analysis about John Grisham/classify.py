"""
classify.py
"""
import operator
import pickle
import string
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import numpy as np
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

def create_csv(tweets):
    data = []
    with open('test_data.csv', 'w', newline='',encoding='utf-8') as fp:
        a = csv.writer(fp, delimiter=',')
        for t in tweets:
            arr = []
            arr.append(0)
            arr.append(t['id'])
            arr.append(t['created_at'])
            arr.append('JohnGrisham')
            arr.append(t['user']['screen_name'])
            arr.append(t['text'])
            data.append(arr)
        a.writerows(data)

def train(X,y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def accuracy(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)    

def do_cross_validation(model, X, y, n_folds):
    cv = KFold(len(y), n_folds)
    accuracies = []
    neutral = 0
    positive = 0
    negative = 0
    for train_ind, test_ind in cv:
        model.fit(X[train_ind], y[train_ind])
        predictions = model.predict(X[test_ind])
        accuracies.append(accuracy(y[test_ind], predictions))
    print('Average 5-fold cross validation accuracy=%.2f (std=%.2f)' %
            (np.mean(accuracies), np.std(accuracies)))

def print_results_to_file(predictions,model,tweets):
    negative = []
    positive = []
    neutral = []
    
    f = open('classify_data_results.txt','a+',encoding='utf-8')
    for i in range(len(predictions)):
        label = predictions[i]
        if label == 0:
            negative.append(tweets.iloc[i]['text'])
        elif label == 2:
            neutral.append(tweets.iloc[i]['text'])
        elif label == 4:
            positive.append(tweets.iloc[i]['text'])
    
    f.write("\nPositive class:%s"%len(positive))
    f.write("\nNegative class:%s"%len(negative))
    f.write("\nNeutral class:%s"%len(neutral))
    
    if len(negative) > 0:
        f.write("\nNegative instance example:")
        f.write(negative[0])
        f.write("\n")
    if len(positive) > 0:
        f.write("\nPositive instance example:")
        f.write(positive[0])
        f.write("\n")
    if len(neutral) > 0:
        f.write("\nNeutral instance example:")
        f.write(neutral[0])
        f.write("\n")
    
    f.close()    
        
def main():
    f = open('classify_data_results.txt','w',encoding='utf-8')
    #tweets = pickle.load(open("tweets.pkl", "rb"))
    #create_csv(tweets[:50])
    tweets_labelled = pd.read_csv('training_labelled.csv',
                     header=None,
                     names=['polarity', 'id', 'date',
                            'query', 'user', 'text'])
    test_data = pd.read_csv('test_data.csv',
                     header=None,
                     names=['polarity', 'id', 'date',
                            'query', 'user', 'text'])
    
    vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1))
    
    list_of_tweets = tweets_labelled['text']
    list_of_test_data = test_data['text']
    X = vectorizer.fit_transform(list_of_tweets)
    y = np.array(tweets_labelled['polarity'])
    
    #training_tweets_df, testing_tweets_df = train_test_split(tweets_labelled)
    #X_train = vectorizer.fit_transform(list_of_tweets)
    vocab = np.array(vectorizer.get_feature_names())
    
    vectorizer = CountVectorizer(min_df=1,ngram_range=(1,1),vocabulary=vocab)
    X_test = vectorizer.fit_transform(list_of_test_data)
    y_test = np.array(test_data['polarity'])
    
    model = train(X,y)
    predictions = model.predict(X_test)
    
    
    print('testing accuracy=%f' %
          accuracy(y_test, predictions))
    
    do_cross_validation(model,X,y,5)
    
    f.close()
    print_results_to_file(predictions,model,test_data)
    
if __name__ == '__main__':
    main()