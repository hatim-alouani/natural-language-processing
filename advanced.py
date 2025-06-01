from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
import spacy

english = spacy.load("en_core_web_sm")

def cleaning(text):
    doc = english(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def bowModel():
    return CountVectorizer()

def TF_IDF_Model():
    return TfidfVectorizer()

def Word2VecModel(sentences):
    return Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

def FastTextModel(sentences):
    return FastText(sentences=sentences, vector_size=100, window=3, min_count=1, workers=4)

text = 'NLP is a fascinating technique! I started learning it 2 days ago.'
text = cleaning(text)

tokens = [text.split()]

cv = bowModel()
tv = TF_IDF_Model()
cbow = Word2VecModel(tokens)
ft = FastTextModel(tokens)

x = cv.fit_transform([text])
print("BoW:\n", x.toarray())
x = tv.fit_transform([text])
print("TF-IDF:\n", x)
text = text.split()
x = cbow.predict_output_word(text, topn=5)
print("Word2vec:\n", x)
x = ft.predict_output_word(text, topn=5)
print("Word2vec:\n", x)