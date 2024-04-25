import pandas as pd
import nltk
from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("steam_reviews.csv")
review_list = df['review'].tolist()

stop_words = set(stopwords.words('english'))

# Add custom stop words
with open("stop_words", "r") as file:
    custom_stop_words = file.read().splitlines()
stop_words.update(custom_stop_words)

# Tokenize & Stemming
tokenizer = TreebankWordTokenizer()
stemmer = PorterStemmer()
stems_list = []
for review in review_list:
    if type(review) is float:
        continue
    tokens = tokenizer.tokenize(review)
    if len(tokens) < 3:
        continue
    stems = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
    stems_list.append(stems)

keyword_list = set()
for stems in stems_list:
    keyword_list.update(stems)

freq_data = []

for stems in stems_list:
    counter = Counter(stems)
    document = {word: counter[word] for word in keyword_list}
    freq_data.append(document)

freq_df = pd.DataFrame(freq_data)

# Frequency
print("\n\n[Frequency]")
keywords_freq = freq_df.sum()
sorted_keywords_freq = keywords_freq.sort_values(ascending=False)
top_10_keywords_freq = sorted_keywords_freq.head(100)
print(top_10_keywords_freq)

# TF-IDF
print("\n\n[TF-IDF]")
tf_dict = freq_df.sum()
tf_idf = {}

for tf_key, tf_value in tf_dict.items():
    idf = (freq_df[tf_key] != 0).sum()
    if idf < 5:
        continue
    tf_idf[tf_key] = tf_value / idf

sorted_dict = sorted(tf_idf.items(), reverse=True, key=lambda x: x[1])

top_10 = sorted_dict[:100]

for key, value in top_10:
    print(key, ":", value)
