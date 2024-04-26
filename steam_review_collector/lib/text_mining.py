import pandas as pd
from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter


def get_stopwords(language):
    # Get prepared stop words
    stopwords_set = set()
    if language != "korean":
        stopwords_set = set(stopwords.words(language))

    # Add custom stop words
    with open("params/stopwords/" + language, "r") as file:
        custom_stop_words = file.read().splitlines()
    stopwords_set.update(custom_stop_words)

    return stopwords_set


def validate_doc(tokens):
    if len(tokens) < 3:
        return False
    return True


def stemming(doc_list, stopwords):
    tokenizer = TreebankWordTokenizer()
    stemmer = PorterStemmer()
    stems_list = []
    for doc in doc_list:
        if (type(doc) == float):
            continue
        tokens = tokenizer.tokenize(doc)
        if not validate_doc(tokens):
            continue
        stems = [stemmer.stem(token) for token in tokens if token.lower() not in stopwords]
        stems_list.append(stems)

    return stems_list


def as_term_frequency_dataframe(stems_list):
    term_list = set()
    for stems in stems_list:
        term_list.update(stems)

    table = []

    for stems in stems_list:
        counter = Counter(stems)
        record = {word: counter[word] for word in term_list}
        table.append(record)

    return pd.DataFrame(table)
