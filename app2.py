import streamlit as st
import numpy as np
import pickle
import pandas as pd
import scikit-learn as sklearn
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load model and scaler once
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('randomFG.pkl', 'rb'))

# Streamlit page configuration
st.set_page_config(page_title='RRP APP', layout='wide', page_icon='image.png')

# Title
st.markdown("<h1 style='text-align: center;'>Restaurant Rating Prediction APP</h1>", unsafe_allow_html=True)
st.caption("This app helps predict restaurant ratings")

# Custom Divider
st.markdown("""
    <hr style="border: none; height: 1px; width: 60%; 
    background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);
    margin: 0 auto;" />
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(Image.open('image.png'), width=200)
    st.title("Made By: MASOOD KHAN")
    
    st.markdown('<a href="mailto:masoodkhanse884@gmail.com" target="_blank">'
                '<button style="background-color:#4CAF50; color:white; padding:10px 20px; border:none; border-radius:5px;">'
                'Contact Us</button></a>', unsafe_allow_html=True)

    st.title("Go To my GitHub")
    st.markdown('<a href="https://github.com/Masoodkhan884" target="_blank">'
                '<button style="background-color:#4CAF50; color:white; padding:10px 20px; border:none; border-radius:5px;">'
                'GitHub</button></a>', unsafe_allow_html=True)

# User Inputs
average_cost = st.number_input("Enter the Average cost for two", min_value=30, max_value=9999, value=1000, step=150)
table_booking = st.selectbox("Restaurant has Table Booking", ["Yes", "No"])
online_delivery = st.selectbox("Restaurant has Online Delivery", ['Yes', 'No'])
price_range = st.selectbox('Select Price Range (1 = Cheapest, 4 = Most Expensive)', [1, 2, 3, 4])

# Preprocess input
values = np.array([[average_cost, int(table_booking == "Yes"), int(online_delivery == "Yes"), price_range]])
X = scaler.transform(values)

# Prediction
if st.button("Predict"):
    prediction = model.predict(X)[0]
    st.balloons()
    
    rating_labels = {
        (0, 2.5): "Poor",
        (2.5, 3.5): "Average",
        (3.5, 4.0): "Good",
        (4.0, 4.5): "Very Good",
        (4.5, 5.0): "Excellent"
    }

    for (low, high), label in rating_labels.items():
        if low <= prediction < high:
            st.markdown(f"<div class='rating-style'>{label}</div>", unsafe_allow_html=True)
            break

# Custom CSS for Styling
st.markdown("""
<style>
.rating-style {
    color: red;
    font-size: 24px;
    font-weight: bold;
    background-color: #ffe6e6;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    border: 2px solid red;
}
</style>
""", unsafe_allow_html=True)
