# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these. 
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", True)
    array(['necronomicon', 'geträumte', 'sünden.<br>hi'], 
          dtype='<U13')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", False)
    array(['necronomicon', 'geträumte', 'sünden', 'br', 'hi'], 
          dtype='<U12')
    """
    results = []
    doc = doc.lower()
    
    if keep_internal_punct:
        results_tokenize = [word.strip(string.punctuation) for word in doc.split(" ")]
        results = [token.strip() for token in results_tokenize if token]
    else:
        doc = re.sub(r'\W+',' ',doc)
        results = doc.strip().split()
    
    return np.hstack(results)
    pass


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    c = Counter(tokens)
    for key,freq in dict(c).items():
        feat_key = "".join(("token=",key))
        feats[feat_key] = freq        
    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    res = [tokens[i:i+k] for i in range(len(tokens)-2)]
    for subset in res:
        res_inner = combinations(subset,2)
        for inner in res_inner:
            key = "".join(("token_pair=",inner[0]+"__"+inner[1]))
            feats[key] = feats[key] + 1 
    pass


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    negative_word_count = 0
    positive_word_count = 0
    feats["neg_words"] = 0
    feats["pos_words"] = 0
    for token in tokens:
        token = token.lower()
        if token in neg_words:
            negative_word_count += 1
            feats["neg_words"] = negative_word_count
        elif token in pos_words:
            positive_word_count += 1
            feats["pos_words"] = positive_word_count
    pass


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    feats = defaultdict(lambda: 0)
    final_feats_result = []
   
    def run_function(func_name,tokens,feats):
        return func_name(tokens,feats)
    
    for func in feature_fns:
        run_function(func,tokens,feats)
    
    for key,value in feats.items():
        final_feats_result.append((key,value))
    
    return sorted(final_feats_result,key=lambda k:k[0]) 
    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    feature_list = []
    data = []
    row = []
    col = []
    count_list = defaultdict(lambda:0) # dict from feature to number of documents in which it occurs
    
    for token in tokens_list:
        feature_list.append(featurize(token,feature_fns))
    
    
    feature_list_flattened = [item for sub_list in feature_list for item in sub_list] #keep duplicate tuples
    count_list = Counter(elem[0] for elem in feature_list_flattened)
    
    d = defaultdict(list)
    for token,num in feature_list_flattened:
        d[token].append(num)
        
    d = {k: sum(v) for k, v in d.items()}    
    feature_list_flat = sorted(d.keys())
    
    if vocab == None:
        vocab = defaultdict(lambda:0)
        index = 0
        for feature in feature_list_flat:
            if count_list[feature] >= min_freq:
                vocab[feature] = index
                index = index + 1
            
    for index,features in enumerate(feature_list):
        for feature in features:
            if feature[0] in vocab.keys():
                data.append(feature[1])
                row.append(index)
                col.append(vocab[feature[0]])
    
    X = csr_matrix((data, (row, col)), dtype=np.int64)
    
    return X,dict(vocab)    
    pass


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    cv = KFold(len(labels), k)
    accuracies = []
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    
    mean_value = np.mean(accuracies)
    
    return mean_value
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    
    accuracy_list = []
    
    count = 0
    for punct in punct_vals:
        tokens = [tokenize(d,punct) for d in docs]
        for minimum_freq in min_freqs:
            for i in range(len(feature_fns),0,-1):
                feature_list = []
                feature_list = list(combinations(feature_fns,i))
                feature_list = [list(elem) for elem in feature_list]
                for func in feature_list:
                    model = LogisticRegression()
                    X,vocab = vectorize(tokens,func,minimum_freq,vocab=None)
                    accuracy_value = cross_validation_accuracy(model,X,labels,5)
 
                    accuracy_dict = defaultdict()
                    accuracy_dict['punct'] = punct
                    accuracy_dict['features'] = tuple(func)
                    accuracy_dict['min_freq'] = minimum_freq
                    accuracy_dict['accuracy'] = accuracy_value
                    accuracy_list.append(accuracy_dict)
                                  
    return sorted(accuracy_list, key=lambda k: k['accuracy'],reverse=True) 
    pass


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    y_values = sorted(d['accuracy'] for d in results)
    x_values = [i for i in range(len(y_values))]
    plt.plot(x_values,y_values)
    plt.xlabel("setting")
    plt.ylabel("accuracy")
    plt.savefig('accuracies.png')
    pass


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    mean_accuracy_setting_list = []
    setting_to_accuracy_dict = defaultdict(list)
    for d in results:
        for k,v in d.items():
            key = ""
            if k != 'accuracy':
                key = "".join((k,"="))
                if k == 'min_freq' or k == 'punct':
                    key = "".join((key,str(v)))
                elif k == 'features':
                    for feature in list(v):
                        key = " ".join((key,feature.__name__))
                                  
            if len(key) != 0:
                setting_to_accuracy_dict[key].append(d['accuracy'])
    
    for k,v in setting_to_accuracy_dict.items():
        mean_value_setting = 0
        mean_value_setting = np.mean(v)
        mean_accuracy_setting_list.append((mean_value_setting,k))
    
    return sorted(mean_accuracy_setting_list, key= lambda k:-k[0])
    pass


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    token_list = [tokenize(d,best_result['punct']) for d in docs]
    X,vocabulary = vectorize(token_list,best_result['features'],best_result['min_freq'],vocab=None)
    clf = LogisticRegression()
    clf.fit(X,labels)
    return clf,vocabulary
    pass


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    positive_sentiment_top_coefficients=[]
    negative_sentiment_top_coefficients=[]
    vocab_key_list = list(vocab.values())
    vocab_inverted = dict((v, k) for k, v in vocab.items())
    for v in vocab_key_list:
        if clf.coef_[0][v] > 0:
            positive_sentiment_top_coefficients.append((vocab_inverted[v],clf.coef_[0][v]))
        else:
            negative_sentiment_top_coefficients.append((vocab_inverted[v],clf.coef_[0][v]))
    
    top_pos = sorted(positive_sentiment_top_coefficients,key = lambda k:-k[1])[:n]
    top_neg = sorted(negative_sentiment_top_coefficients,key = lambda k:k[1])[:n]
    top_neg = [(value[0],abs(value[1])) for value in top_neg]

    if label == 1:
        return top_pos
    elif label == 0:
        return top_neg
    pass


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    
    test_docs,test_labels = read_data(os.path.join('data', 'test'))
    token_list_test = [tokenize(d,best_result['punct']) for d in test_docs]
    X_test,vocab = vectorize(token_list_test,best_result['features'],best_result['min_freq'],vocab)
    
    return test_docs,test_labels,X_test
    pass


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    predictions = clf.predict(X_test)
    predictions_prob = clf.predict_proba(X_test)
    probs = {}
    for i in range(len(predictions)):
        label = predictions[i]
        if predictions[i] != test_labels[i]:
            probs[i] = predictions_prob[i][label]
              
    for index in sorted(probs, key=probs.get, reverse=True)[:n]:               
        print("\ntruth=%s predicted= %s proba=%s"%(test_labels[index],predictions[index],probs[index]))
        print(test_docs[index])
    
    pass


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    
    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
