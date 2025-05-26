import nltk
import re
import numpy as np 
import pandas as pd 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

def tokenization_sentences(text):
    return sent_tokenize(text)

def tokenization_words(text):
    return word_tokenize(text)

def stop_words(text):
    return [token for token in tokens if token not in set(stopwords.words('english'))]

def cleaning(text):
    text = re.sub('[a^zA^Z]', ' ', text)
    text = text.lower().split()
    return ' '.join(text)

def stemming(tokens):
    return [PorterStemmer().stem(token) for token in tokens]

def lemmatization(tokens):
    return [WordNetLemmatizer().lemmatize(token) for token in tokens]

def bow(text):
    cv = CountVectorizer()
    x = cv.fit_transform(text).toarray()
    return x
