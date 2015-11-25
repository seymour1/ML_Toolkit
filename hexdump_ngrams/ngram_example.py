import nltk
from nltk import bigrams, trigrams
from sklearn.feature_extraction.text import TfidfVectorizer

with open("hexdump",'r') as file:
    bytes = file.read()
    byte_tokens = nltk.regexp_tokenize(bytes, pattern='\w\w')

byte_bigrams = bigrams(byte_tokens)
byte_trigrams = trigrams(byte_tokens)

vectorizer = TfidfVectorizer(min_df=2)
X = vectorizer.fit_transform(''.join(byte_bigram) for byte_bigram in byte_bigrams)
idf = vectorizer.idf_

tf_idfs = dict(zip(vectorizer.get_feature_names(), idf))
for bytes in tf_idfs:
    print bytes, tf_idfs[bytes]
