import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

def stop_words(text):
    text = [word for word in text if word.lower() not in set(stopwords.words('english'))]
    return text

def tokenization_sentences(text):
    return sent_tokenize(text)

def tokenization_words(text):
    return word_tokenize(text)

def lemmatization(tokens):
    return [WordNetLemmatizer().lemmatize(token) for token in tokens]

def stemming(tokens):
    return [PorterStemmer().stem(token) for token in tokens]

def cleaning(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = ' '.join(text)
    return text

text = "nlp is amazing"
text = cleaning(text)
tokens = tokenization_words(text)
tokens = stop_words(tokens)
tokens = lemmatization(tokens)
text = ' '.join(tokens)

cv = CountVectorizer()
tv = TfidfVectorizer()
x_cv = cv.fit_transform([text])
x_tv = tv.fit_transform([text])


tokens = [tokenization_words(text)]

print(text)
print(tokens)

# tf = FastText(sentences=tokens)
# WV = Word2Vec(sentences=tokens)
