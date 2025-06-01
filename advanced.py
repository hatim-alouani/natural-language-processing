from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors, FastText
import spacyPreProcessing
from spacyPreProcessing import cleaning

def bow():
    cv = CountVectorizer()
    return cv

def TF_IDF():
    tv = TfidfVectorizer()
    return tv
    
def Word2Vec(sentences):
    cbow =  Word2Vec(sentences=sentences)
    return cbow

def FastTextModel(sentences):
    ft = FastText(sentences=sentences, vector_size=200, window=3, min_count=1, workers=4)
    return ft

text = english('NLP is a fascinating technique! I started learning it 2 days ago.')
text = cleaning(text)

cv = bow()
tv = TF_IDF()
cbow = Word2Vec(tokens)
tf = FastText(tokens)

x = cv.fit_transform(tokens)
print(x.toarray())
x = tv.fit_transform(tokens)
print(x.toarray())
x = cbow.predict(tokens)
print(x.toarray())
# x = 
# print(x.toarray())