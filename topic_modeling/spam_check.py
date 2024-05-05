import pandas as pd

term_list = pd.read_csv('data/tf.csv')['term'].tolist()
review_list = pd.read_csv('review/bluearchive.csv')['review'].tolist()

spam = []
for review in review_list:
    for term in term_list:
        if type(review) is float:
            continue
        if review.count(term) > 10:
            spam.append({term: review})
            break

for s in spam:
    print(s)