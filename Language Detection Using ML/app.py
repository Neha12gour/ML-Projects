import streamlit as st
import pickle


cv =pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title("Language Detector")
input_text = st.text_area("Enter your text in any Language")

if st.button("Predict"):

    #1. vectorize
    vector_input= cv.transform([input_text])

    #2. predict
    result = model.predict(vector_input)[0]

    #3. display
    st.header('The input Language is in '+ result)