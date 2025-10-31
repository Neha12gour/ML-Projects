import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



#stemming
ps=PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]'," ",content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

tfidf =pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))



#website
st.title("Fake News Detector")
input_text= st.text_area("Enter the message")


if st.button("Predict"):
    #1. preprocess
    transform_text= stemming(input_text)

    #2. vectorize
    vector_input= tfidf.transform([transform_text])

    #3. predict
    result = model.predict(vector_input)[0]

    #4. display
    if result == 1:
        st.header("Fake News")
    else:
        st.header("Real News")

