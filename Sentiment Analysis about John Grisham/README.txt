The intent of this project is to collect data related to best selling author John Grisham and analyse the sentiment about him and his novels. 

First, in collect.py, we collect around 400 tweets about John Grisham and store these tweets in a pickle file. Then we iterate over the list of tweets and 
keep track of the profiles of the users who have written these tweets.
Then we find the friends of these users and track those profiles or users who are following John Grisham. 

In cluster.py we form a network between John Grisham and all those who have tweeted about him and their friends who also follow John Grisham. 
After forming around 4 communities based on edge betweenness,
we observe that the communities are majorly John Grisham and all his followers/users who tweet about him, 
other authors who write same genre as him or are as successful as him.

In classify.py we train using logistic regression by manually labelling sentiment for about 400 odd tweets and then test on about 50 tweets. It is noticed that
overall the twitter audience has a very positive(~47.5%) or neutral sentiment(~50%) towards John Grisham. There is very very less negative sentiment(~3%). 
This is understandable as he is a well-known, best selling author and a loved celebrity and he has not been involved in any major controversy.

In summarize.py we collate all of the above information.  

