"""
    스팸 리뷰를 찾아내기 위한 스크립트
    미리 tf 파일이 생성되어 있어야 합니다
"""

import pandas as pd
from tqdm import tqdm

TF_FILE = "data/tf.csv"
REVIEW_FILE = 'review/starrail.csv'

term_list = pd.read_csv(TF_FILE)['term'].tolist()
review_list = pd.read_csv(REVIEW_FILE)['review'].tolist()

spam_reviews = []
for review in tqdm(review_list, '도배 의심 리뷰 색출'):
    for term in term_list:
        if type(review) is float:  # 비어 있는 리뷰 제외
            break
        if review.count(term) > 10:
            spam_reviews.append({term: review})  # 동일한 Term이 10번 이상 등장하면 도배 의심
            break

print("\n다음 리뷰들은 스팸 리뷰으로 의심됩니다...\n")
for spam_review in spam_reviews:
    print(spam_review)
