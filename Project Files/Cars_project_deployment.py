#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pickle


# In[2]:


xgb_model=pickle.load(open('Cars_project.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
feature_columns = pickle.load(open('feature_columns.pkl','rb'))


# In[3]:


st.title('Cars24 Model Deployment')


# In[4]:


def user_input_parameter():

    year = st.sidebar.number_input('Year', 2000, 2025, 2019)
    kilometerdriven = st.sidebar.number_input('Kilometers Driven',min_value=0)
    ownernumber = st.sidebar.selectbox('Owner Number',[1,2,3,4])
    isc24assured = st.sidebar.selectbox('C24 Assured',[0,1])
    benefits = st.sidebar.number_input('Benefits',min_value=0)
    discountprice = st.sidebar.number_input('Discount Price',min_value=0)
    make = st.sidebar.selectbox('Make',['Hyundai','Maruti','Honda','Toyota','Tata','Mahindra','Kia'])
    fueltype = st.sidebar.selectbox('Fuel Type',['Diesel','Petrol','Petrol + Cng'])
    transmission = st.sidebar.selectbox('Transmission',['Automatic','Manual'])
    bodytype = st.sidebar.selectbox('Body Type',['SUV','Sedan','Hatchback'])
    city = st.sidebar.selectbox('City',['Hyderabad','Bangalore','Mumbai','Chennai','Kolkata'])
    car_age = 2026 - year

    dict1 = {
        'year': year,
        'kilometerdriven': kilometerdriven,
        'ownernumber': ownernumber,
        'isc24assured': isc24assured,
        'benefits': benefits,
        'discountprice': discountprice,
        'car_age': car_age,
        'make': make,
        'fueltype': fueltype,
        'transmission': transmission,
        'bodytype': bodytype,
        'city': city
    }

    features = pd.DataFrame(dict1, index=[0])

    return features

df = user_input_parameter()

button = st.button('Predict Price')

if button:
    df = pd.get_dummies(df)

    df = df.reindex(columns=feature_columns,fill_value=0)

    df_scaled = scaler.transform(df)

    pred = xgb_model.predict(df_scaled)

    st.subheader('Predicted Price')

    st.write(f'₹ {pred[0]:,.0f}')

