"""
collect.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import itertools
from TwitterAPI import TwitterAPI
import pickle

consumer_key = '1rhjWHiOG0xfHf5zkQL3yqjK6'
consumer_secret = 'qWxOCAgtZbhgXb8uJelTPE4xdlAm9A9N64vK9XvRTuCv4Stm2w'
access_token = '768548145852669952-qDKfAdGqT2gZZXgTFxQqy2w3rSjzf2U'
access_token_secret = 'b9JoEo88LoauZpFP3Twny6raogOjGvHeXv7rAMvCz580F'
friend_dict = {}

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        try:
            request = twitter.request(resource, params)
            if request.status_code == 200:
                return request
            else:
                print('Got error %s \nsleeping for 15 minutes.' % request.text)
                sys.stderr.flush()
                time.sleep(61 * 15)
        except:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            time.sleep(61 * 15)

def gather_tweets():
    twitter = get_twitter()
    request = robust_request(twitter,'search/tweets', {'q':'@JohnGrisham','language':'en'})
    tweets = [r for r in request]
    
    maximum_id = tweets[len(tweets)-1]['id']
    while len(tweets) <= 400:
        request = robust_request(twitter,'search/tweets', {'q':'@JohnGrisham','language':'en','max_id':maximum_id})
        for r in request:
            tweets.append(r)
        maximum_id = tweets[len(tweets)-1]['id']
    
    pickle.dump(tweets, open('tweets.pkl', 'wb'))
    f = open('collect_data_results.txt','w',encoding='utf-8')
    number_of_users = len(set([t['user']['screen_name'] for t in tweets]))
    number_of_messages = len(tweets)
    f.write("Number of users collected:%s"%number_of_users)
    f.write("\nNumber of messages collected:%s"%number_of_messages)
    f.close()
    return tweets

def get_friends(user_screen_name):
    twitter = get_twitter()
    
    
    request = robust_request(twitter,'friends/list',{'screen_name':user_screen_name,'count':200})
    friends = [r for r in request]
    friend_dict[user_screen_name] = friends
    pickle.dump(friend_dict, open('friends.pkl', 'wb'))
    return friends

def get_followers():
    twitter = get_twitter()
    request = robust_request(twitter,'followers/list',{'screen_name':'JohnGrisham','count':200})
    followers = [r for r in request]
    pickle.dump(followers, open('followers.pkl', 'wb'))
    return followers
    
def main():
    tweets = gather_tweets()
    tweet_authors = [t['user']['screen_name'] for t in tweets]
    get_followers()
    for author in tweet_authors:
        get_friends(author)

if __name__ == '__main__':
    main()


