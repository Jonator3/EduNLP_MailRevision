import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import transformers
import gst_calculation


def levenshtein_distance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


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
    for i1 in range(len(text1)):
        for i2 in range(len(text2)):
            l = __equal_till(text1[i1:], text2[i2:])
            lcs = max(lcs, l)
    return lcs


def longest_common_tokensubstring(text1, text2, lemmatize=False):
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return longest_common_substring(tokens1, tokens2)


def gst(text1, text2, minmatch=3):
    _, score = gst_calculation.gst.calculate(text1, text2, minimal_match=minmatch)
    return score


def token_gst(text1, text2, minmatch=3, lemmatize=False):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return gst(tokens1, tokens2, minmatch)


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

    return similarity[0][0]


def bert_vector_cosine(text1, text2):
    # Load the BERT model
    model = transformers.BertModel.from_pretrained('bert-base-uncased')

    # Tokenize and encode the texts
    encoding1 = model.encode(text1, max_length=512)
    encoding2 = model.encode(text2, max_length=512)

    # Calculate the cosine similarity between the embeddings
    similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
    return similarity
