"""
streamlit_app.py
------------------
Streamlit version of the House Price Prediction app.
Uses the SAME trained model (model/house_price_model.pkl) that
train_model.py produces - no retraining needed here.

Run locally with:
    streamlit run streamlit_app.py

Deploy for free on Streamlit Community Cloud (see README section
"Deploying to Streamlit" for step-by-step instructions).
"""

import streamlit as st
import numpy as np
import joblib
import os

# -----------------------------------------------------------------
# Page config (must be the first Streamlit call)
# -----------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------------------------------------------
# Load the trained model (cached so it only loads once per session)
# -----------------------------------------------------------------
FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "Age", "Parking"]
MODEL_PATH = os.path.join("model", "house_price_model.pkl")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found! Please run `python train_model.py` "
            "first to train and save the model."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()


def format_indian_currency(amount):
    """Formats a number into the Indian numbering system, e.g. 1234567 -> '12,34,567'."""
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return s
    last_three = s[-3:]
    remaining = s[:-3]
    parts = []
    while len(remaining) > 2:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.insert(0, remaining)
    return ",".join(parts) + "," + last_three


# -----------------------------------------------------------------
# UI
# -----------------------------------------------------------------
st.title("🏠 House Price Prediction")
st.write(
    "A simple Linear Regression model that estimates a house's price "
    "based on its features. Enter the details below and click **Predict**."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq.ft)", min_value=100.0, max_value=20000.0, value=1500.0, step=50.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2, step=1)

with col2:
    floors = st.number_input("Floors", min_value=0, max_value=10, value=2, step=1)
    age = st.number_input("Age of house (years)", min_value=0, max_value=200, value=5, step=1)
    parking = st.number_input("Parking spaces", min_value=0, max_value=10, value=1, step=1)

st.divider()

if st.button("Predict House Price", type="primary", use_container_width=True):
    input_array = np.array([[area, bedrooms, bathrooms, floors, age, parking]])
    predicted_price = model.predict(input_array)[0]
    formatted_price = format_indian_currency(predicted_price)

    st.success("Prediction complete!")
    st.metric(label="Estimated House Price", value=f"₹ {formatted_price}")

    with st.expander("See the values you entered"):
        st.table({
            "Feature": FEATURES,
            "Value": [area, bedrooms, bathrooms, floors, age, parking]
        })

st.caption("Built with Streamlit & Scikit-learn | College ML Assignment")
