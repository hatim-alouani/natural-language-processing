import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text = 'NLP is a fascinating technique! I started learning it 2 days ago.'

def tokenization_sentences(text):
    return sent_tokenize(text)

def tokenization_words(text):
    return word_tokenize(text)

def stop_words(tokens):
    return [token for token in tokens if token.lower() not in set(stopwords.words('english'))]

def cleaning(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join(text)

def stemming(tokens):
    return [PorterStemmer().stem(token) for token in tokens]

def lemmatization(tokens):
    return [WordNetLemmatizer().lemmatize(token) for token in tokens]

def BagOfWords(text):
    cv = CountVectorizer()
    x = cv.fit_transform([text]).toarray()
    return x

def TF_IDF(text):
    tv = TfidfVectorizer()
    x = tv.fit_transform([text]).toarray()
    return x

def Word2VecModel(sentences):
    cbow = Word2Vec(sentences=sentences, vector_size=200, window=3, min_count=1, workers=4, sg=0)
    return cbow

def FastTextModel(sentences):
    fasttext = FastText(sentences=sentences, vector_size=200, window=3, min_count=1, workers=4)
    return fasttext
