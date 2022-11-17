from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import nltk
import nltk
import pickle
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))


def clean_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


def transform(text):
    seq = tokenizer.texts_to_sequences(text)
    return seq
