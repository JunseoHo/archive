"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   사용자 지정 변수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd

# 리뷰 데이터가 저장되어 있는 csv 파일의 위치
PREFIX = "negative_"

IN_FILE_PATH = "data/" + PREFIX + "reviews.csv"

# 외부에 출력할 파일 이름
TF_FILE_PATH = "data/" + PREFIX + "tf.csv"
TF_TOP100_FILE_PATH = "data/" + PREFIX + "tf_top100.csv"
TF_IDF_FILE_PATH = "data/" + PREFIX + "tf_idf.csv"
CO_OCURR_TOP100_FILE_PATH = "data/" + PREFIX + "co_occur.csv"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   메인 스크립트
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
csv = pd.read_csv(IN_FILE_PATH)
review_list = csv['review'].tolist()

keywords = ["개발자", "업데이트", "한글"]

for review in review_list:
    if isinstance(review, float):
        continue
    if all(keyword in review for keyword in keywords):
        emphasized_review = ""

        for idx, char in enumerate(review):
            if idx % 30 == 0 and idx != 0:
                emphasized_review += "\n"  # 30글자마다 줄바꿈
            emphasized_review += char
        print(emphasized_review)
        print('\n\n')
