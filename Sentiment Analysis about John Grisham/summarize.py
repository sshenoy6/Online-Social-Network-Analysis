import collect
import cluster
import numpy as np
import pickle
import classify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import os.path 
"""
sumarize.py
"""
def main():

    f = open('summary.txt','w',encoding='utf-8')
    if(os.path.isfile('collect_data_results.txt')):
        f_collect = open('collect_data_results.txt','r')
        for line in f_collect.readlines():
            f.write(line)
        f_collect.close()
    else:
        tweets = pickle.load(open("tweets.pkl", "rb"))
        number_of_users = len(set([t['user']['screen_name'] for t in tweets]))
        number_of_messages = len(tweets)
        f.write("Number of users collected:%s"%number_of_users)
        f.write("\nNumber of messages collected:%s"%number_of_messages)
    
    f_cluster = open('cluster_data_results.txt','r',encoding='utf-8')
    for line in f_cluster.readlines():
        f.write(line)
    f_cluster.close()
    
    f_classify = open('classify_data_results.txt','r',encoding='utf-8')
    for line in f_classify.readlines():
        f.write(line)
    f_classify.close()
    
    f.close()
    
if __name__ == '__main__':
    main()
    