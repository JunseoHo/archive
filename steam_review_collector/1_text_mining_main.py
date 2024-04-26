from lib.text_mining import *

df = pd.read_csv("steam_reviews.csv")
review_list = df['review'].tolist()

stopwords = get_stopwords("korean")
stems_list = stemming(review_list, stopwords)
tf_df = as_term_frequency_dataframe(stems_list)

# Frequency
print("\n\n[Frequency]")
keywords_freq = tf_df.sum()
sorted_keywords_freq = keywords_freq.sort_values(ascending=False)
top_10_keywords_freq = sorted_keywords_freq.head(100)
print(top_10_keywords_freq)

# TF-IDF
print("\n\n[TF-IDF]")
tf_dict = tf_df.sum()
tf_idf = {}

for tf_key, tf_value in tf_dict.items():
    idf = (tf_df[tf_key] != 0).sum()
    if idf < 5:
        continue
    tf_idf[tf_key] = tf_value / idf

sorted_dict = sorted(tf_idf.items(), reverse=True, key=lambda x: x[1])

top_10 = sorted_dict[:100]

for key, value in top_10:
    print(key, ":", value)
