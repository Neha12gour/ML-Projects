import streamlit as st
import helper
import pickle

model = pickle.load(open('model.pkl', 'rb'))

st.header("Duplicate Question Pair")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Find"):

    #for exact match
    if q1.strip().lower() == q2.strip().lower():
        st.header("Duplicate Question")

    else:
        query = helper.query_point_creator(q1, q2)
        result = model.predict(query)[0]

        if result:
            st.header("Duplicate Questions")
        else:
            st.header("Not a Duplicate Questions")