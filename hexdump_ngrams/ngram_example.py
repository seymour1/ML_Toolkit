import nltk
from nltk import bigrams, trigrams
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams

with open("hexdump",'r') as file:
    bytes = file.read()
    byte_tokens = nltk.regexp_tokenize(bytes, pattern='\w\w')

byte_fourgrams = ngrams(byte_tokens, 4)

vectorizer = TfidfVectorizer(min_df=3)
X = vectorizer.fit_transform(''.join(byte_fourgram) for byte_fourgram in byte_fourgrams)
idf = vectorizer.idf_

tf_idfs = dict(zip(vectorizer.get_feature_names(), idf))
for bytes in tf_idfs:
    print bytes, bytes.decode("hex").replace("\n"," "), tf_idfs[bytes]

