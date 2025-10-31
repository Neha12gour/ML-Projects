import streamlit as st
import pickle
import numpy as np
import math



#import the model
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

st.title("Car Price Predictor")

#name
name = st.selectbox('Model', df['name'].unique())

#company
company = st.selectbox('Brand', df['company'].unique())

#year
year = st.selectbox('Brand', df['year'].unique())

#kilo-meter driven
kms_driven = st.number_input('KMS Driven')

#fuel type
fuel_type = st.selectbox('Fuel Type', df['fuel_type'].unique())


if st.button('Predict Price'):
    query = np.array([name, company, year, kms_driven, fuel_type])
    query = query.reshape(1,5)
    st.title("The Predicted price is:  " + str(int(pipe.predict(query)[0])))
