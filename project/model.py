import nltk
import re
import numpy as np 
import pandas as pd 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, BatchNormalization, Dropout, SimpleRNN, GRU, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer

def stop_words(tokens):
    return [token for token in tokens if token not in set(stopwords.words('english'))]

def stemming(tokens):
    return [PorterStemmer().stem(token) for token in tokens]

def lemmatization(tokens):
    return [WordNetLemmatizer().lemmatize(token) for token in tokens]

def cleaning(dataset):
    reviews = []
    for i in range(0, 6000):
      review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
      # review = re.sub(r'\@\w+|\#', '', review)
      review = re.sub(r'[^A-Za-z0-9 ]+', '', review)
      review = review.lower().split()
      review = stop_words(review)
      review = lemmatization(review)
      review =  ' '.join(review)
      reviews.append(review)
    return reviews

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

dataset = pd.read_csv('covid-19_vaccine_tweets_with_sentiment.csv')
reviews = cleaning(dataset)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)
x = tokenizer.texts_to_sequences(reviews)
x = pad_sequences(x, 100)
y = dataset['sentiment'].astype('category').cat.codes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

num_classes = len(dataset['sentiment'].astype('category').cat.categories)
print(num_classes)

model = Sequential([
  Embedding(input_dim=10000, output_dim=64, input_length=100),
  Bidirectional(LSTM(32, activation='tanh', dropout=0.5, recurrent_dropout=0.3)),
  BatchNormalization(),
  Dropout(0.3),
  Dense(64, activation='relu'),
  Dense(units=3, activation='softmax', kernel_regularizer=l2(0.01))
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, batch_size=64, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("test_accuracy : ", test_accuracy)
print("test_loss : ", test_loss)

prediction = model.predict(x_test[125:126])
prediction = np.argmax(prediction)
print("prediction : " ,prediction)
print("true result : ", y_test.iloc[125:126])


print(history.history)