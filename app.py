import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.header("Plagarism-Detector")

def converter(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    stopwords_set = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    text = " ".join(stemmer.stem(word) for word in text.split() if word not in stopwords_set)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    vectors = tokenizer.texts_to_sequences(text)
    vectors = pad_sequences(vectors, padding = "post")
    return vectors

text = st.text_input("Enter the sentence: ")
vectors = converter(text)

model = load_model()


