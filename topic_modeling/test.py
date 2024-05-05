from nltk import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize("콜라보레 이션을 합시다"))