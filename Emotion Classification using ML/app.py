import streamlit as st
import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re



#load models
model = pickle.load(open('logistic_regression.pkl','rb'))
lb = pickle.load(open('label_encoder.pkl','rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))





#===================  custom function =============================

#stemming
def clean_text(content):
    ps = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]'," ",content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content




def prediction(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf.transform([cleaned_text])

    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(model.predict(input_vectorized)[0])
    return predicted_emotion, label





#app===========================================================
st.title("Six NLP Emotions Detection APP")
st.write(["Anger", "Fear", "Joy", "Love", "Sadness", "Surprise"])
input_text = st.text_input("Enter your text")




if st.button("Predict"):
    predicted_emotion, label = prediction(input_text)
    st.write("Predicted Emotion : ", predicted_emotion)
    st.write("Predicted Label : ", label)

