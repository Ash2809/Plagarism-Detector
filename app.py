import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.header("Plagarism-Detector")

def converter(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    stopwords_set = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    text1 = " ".join(stemmer.stem(word) for word in text.split() if word not in stopwords_set)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text1)
    vectors = tokenizer.texts_to_sequences(text1)
    vectors = pad_sequences(vectors, padding = "post")
    return vectors




if __name__ == "__main__":
    print("ok")
    # text = st.text_input("Enter the sentence: ")
    text = "Hello i am aashutosh"
    if text:
        vectors = converter(st.text_input("Enter the sentence: "))
        model = load_model(r"C:\Projects\Plagarism-Detector\models\LSTM_model.h5")
        result = model.predict(vectors)
        if result[0][0] >= 0.5:
            st.write("Plagarized")
            print('yes')
        else:
            st.write("Original")
            print("no")
