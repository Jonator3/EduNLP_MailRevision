import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import transformers
import gst_calculation
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


def length_difference(text1, text2):
    diff = abs(len(text1) - len(text2))
    return diff, diff/max(len(text1), len(text2)), []


def levenshtein_distance(token1, token2):

    distances = np.zeros((len(token1) + 1, len(token2) + 1))
    pre_step = np.zeros((len(token1) + 1, len(token2) + 1))
    diffs = {
        0: (-1, -1, lambda x, y: []),
        1: ( 0, -1, lambda x, y: [(1, y, y+1, 0)]),
        2: (-1,  0, lambda x, y: [(0, x, x+1, 1)]),
        3: (-1, -1, lambda x, y: [(0, x, x+1, 2), (1, y, y+1, 2)])
    }

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1
        pre_step[t1][0] = 2
    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        pre_step[0][t2] = 1
    pre_step[0][0] = -1

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                # no change
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
                pre_step[t1][t2] = 0
            else:
                add = distances[t1][t2 - 1]
                remove = distances[t1 - 1][t2]
                change = distances[t1 - 1][t2 - 1]

                if (add <= remove and add <= change):
                    distances[t1][t2] = add + 1
                    pre_step[t1][t2] = 1
                elif (remove <= add and remove <= change):
                    distances[t1][t2] = remove + 1
                    pre_step[t1][t2] = 2
                else:
                    distances[t1][t2] = change + 1
                    pre_step[t1][t2] = 3

    out = int(distances[len(token1)][len(token2)])
    markings = []
    x = len(token1)
    y = len(token2)
    while True:
        map = diffs.get(int(pre_step[x][y]))
        if map is None:
            break
        xoff, yoff, marker = map
        x += xoff
        y += yoff
        markings += marker(x, y)

    return out, out/max(len(token1), len(token2)), markings


def token_levenshtein_distance(text1, text2, lemmatize=False):
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return levenshtein_distance(tokens1, tokens2)


def __equal_till(s1, s2):
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return i
    return min(len(s1), len(s2))


def longest_common_substring(text1, text2):
    lcs = 0
    start_t1 = 0
    start_t2 = 0
    for i1 in range(len(text1)):
        for i2 in range(len(text2)):
            l = __equal_till(text1[i1:], text2[i2:])
            if l > lcs:
                lcs = l
                start_t1 = i1
                start_t2 = i2
    return lcs, lcs/max(len(text1), len(text2)), [(0, start_t1, start_t1+lcs, 0), (1, start_t2, start_t2+lcs, 0)]


def longest_common_tokensubstring(text1, text2, lemmatize=False):
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return longest_common_substring(tokens1, tokens2)


def gst(string1, string2):
    res = gst_calculation.gst.calculate(string1, string2)
    length = res[-1]
    marking = []
    for section in res[0]:
        t1 = section.get("token_1_position")
        t2 = section.get("token_2_position")
        marking.append((0, t1, t1+section.get("length"), 0))
        marking.append((1, t2, t2+section.get("length"), 0))

    return length, length/max(len(string1), len(string2)), marking


def token_gst(text1, text2, lemmatize=False):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return gst(tokens1, tokens2)


def vector_cosine(text1, text2, lemmatize=False):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectorizer.fit(tokens1+tokens2)

    vector1 = vectorizer.transform(tokens1)
    vector2 = vectorizer.transform(tokens2)
    vector1 = np.asarray(vector1.sum(axis=0)[0])
    vector2 = np.asarray(vector2.sum(axis=0)[0])

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity[0][0], similarity[0][0], []

