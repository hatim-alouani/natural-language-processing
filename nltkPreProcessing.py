import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

def tokenization_words(text):
    return word_tokenize(text)

def tokenization_sentences(text):
    return sent_tokenizer(text)

def lemmatization(tokens):
    return [WordNetLemmatizer().lemmatize(token) for token in tokens]

def stop_words(tokens):
    return [token for token in tokens if token.lower() not in stopwords.words('english')]

def cleaning(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = tokenization_words(text)
    tokens = stop_words(tokens)
    tokens = lemmatization(tokens)
    text = ' '.join(tokens)
    text = text.lower().split()
    text = ' '.join(text)
    return text

text = "NLP is a fascinating technique! I started learning it 2 days ago."
text = cleaning(text)
print(text)
