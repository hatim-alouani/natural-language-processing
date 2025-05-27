import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors, FastText

english = spacy.load("en_core_web_sm") #english library
text = english('NLP is a fascinating technique! I started learning it 2 days ago.')

def tokenization_infos(text):
    for token in text:
        print(token.text)
        print(token.shape)
        print(token.is_alpha)
        print(token.is_stop)
        print(token.like_url)
        print(token.like_email)
        print('-------------')

def tokenization_words(text):
    for token in text:
        print(token.text, end=' | ')

def tokenization_sentences(text):
    return [token.text for token in text.sents]

def lemmatization(text):
    for token in text:
        print(token.text, '\t', token.lemma_)

def grammatical_role(text):
    for token in text:
       print(token.text, '\t', token.pos_) #It helps identify the grammatical role of each word in a sentence

def stop_words(text):
    tokens = [token.text for token in text if not token.is_stop]
    return tokens

def cleaning(text):
    tokens = [token.text for token in text if token.is_alpha]
    text = ' '.join(tokens)
    return english(text)

def bow(text):
    cv = CountVectorizer()
    x = cv.fit_transform(text).toarray()
    return x

def TF_IDF(test):
    tv = TfidfVectorizer()
    x = tf.fit_transform(text)

def Word2Vec(tokens):
    cbow= Word2Vec(sentences=tokens, vector_size=200, window=3, min_count=1, workers=4, sg=0)
    return cbow
