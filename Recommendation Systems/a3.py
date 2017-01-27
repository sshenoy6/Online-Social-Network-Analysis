# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    movies_tokens = []
    for index,row in movies.iterrows():
        genre_string_list = tokenize_string(row['genres'])
        movies_tokens.append(genre_string_list)
    
    movies['tokens'] = np.array(movies_tokens)
    
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    vocab = defaultdict(int)
    term_frequency = defaultdict(int)
    feature_list = []
    count = 0
    list_csr_matrix = []
    list_of_list_of_features = []
    document_frequency = Counter()
    
    for index,row in movies.iterrows():
        tokens_list = sorted(set(row['tokens']))
        list_of_list_of_features.append(row['tokens'])
        for index1,feature in enumerate(tokens_list):
            feature_list.append(feature)
    
    for list_of_features in list_of_list_of_features: 
            document_frequency.update(set(list_of_features))
    
    for index,feature in enumerate(sorted(set(feature_list))):
        vocab[feature] = index
    
    for index,row in movies.iterrows():
        term_count = Counter()
        term_count.update(row['tokens'])   
        data = []
        row_csr_matrix = []
        col_csr_matrix = []
        
        tokens_list = sorted(set(row['tokens']))
        for index1,feature in enumerate(tokens_list):
            term_frequency[feature] = term_count[feature]/term_count.most_common(1)[0][1]
            w = term_frequency[feature] * math.log10((len(movies.index)/document_frequency[feature]))
            col_csr_matrix.append(vocab[feature])
            row_csr_matrix.append(0)
            data.append(w)
                        
        list_csr_matrix.append(csr_matrix((data, (row_csr_matrix, col_csr_matrix)), shape=(1, len(vocab.values()))))
        
    movies['features'] = np.array(list_csr_matrix)
    return (movies,vocab)
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    cosine_similarity_value = a.T.dot(b).sum(axis=0)/(math.sqrt((a.data ** 2).sum(axis=0)) * math.sqrt((b.data ** 2).sum(axis=0)))
    return cosine_similarity_value.sum() 
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    list_of_ratings = []
    for index,movie_to_be_rated in ratings_test.iterrows():
        weighted_average = 0
        cosine_similarity_score_list = []
        average = 0
        for index1,movie_rated in ratings_train[ratings_train.userId==movie_to_be_rated['userId']].iterrows():
            matrix_a = movies[movies.movieId==movie_to_be_rated['movieId']].iloc[0]['features']
            matrix_b = movies[movies.movieId==movie_rated['movieId']].iloc[0]['features']
            
            cosine_similarity_score = cosine_sim(matrix_a,matrix_b)
            if cosine_similarity_score >= 0:
                weighted_average += cosine_similarity_score * movie_rated['rating']
                cosine_similarity_score_list.append(cosine_similarity_score)
            else:
                weighted_average += sum(movie_rated['rating'])
                cosine_similarity_score_list.append(len(ratings_train[ratings_train.userId==movie_to_be_rated['userId']].index))
        
        average = weighted_average/sum(cosine_similarity_score_list)
        list_of_ratings.append(average)
        
    rating = np.array(list_of_ratings)
    return rating
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
