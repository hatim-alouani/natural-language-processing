import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText

def cleaning(text):
    return [token.lemma_.lower() for token in text if token.is_alpha and not token.is_stop]

nlp = spacy.load("en_core_web_sm")
text = 'NLP is a fascinating technique! I started learning it 2 days ago.'
text = nlp(text)
text = cleaning(text)

tokens = [text]

cv = CountVectorizer()
tv = TfidfVectorizer()
x_cv = cv.fit_transform(text)
x_tv = tv.fit_transform(text)


print(text.text)
print(tokens)

tf = FastText(sentences=text)
wv = Word2Vec(sentences=text)
