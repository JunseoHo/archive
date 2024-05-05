from konlpy.tag import Komoran
from nltk import TreebankWordTokenizer

kom = Komoran()
print(kom.pos("무자본"))