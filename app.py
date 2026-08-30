"""
app.py
-------
Flask web application for House Price Prediction.

This app:
1. Loads the trained Linear Regression model at startup.
2. Shows a home page with project info.
3. Shows a form to enter house details.
4. Validates the user's input.
5. Uses the model to predict the price.
6. Displays the result on a result page.

Run with:
    python app.py
Then open http://127.0.0.1:5000/ in your browser.
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ---------------------------------------------------------------
# Load the trained model once, when the server starts.
# ---------------------------------------------------------------
MODEL_PATH = os.path.join("model", "house_price_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found! Please run 'python train_model.py' first "
        "to train and save the model before starting the Flask app."
    )

model = joblib.load(MODEL_PATH)

# The order of features MUST match the order used during training.
FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "Age", "Parking"]


@app.route("/")
def home():
    """Home page with project description."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  -> show the empty prediction form
    POST -> read form values, validate them, predict price, show result
    """
    if request.method == "GET":
        return render_template("index.html", show_form=True)

    # ---- POST request: process the form ----
    errors = []
    values = {}

    for feature in FEATURES:
        raw_value = request.form.get(feature, "").strip()
        try:
            # All our features are numeric (int or float)
            value = float(raw_value)
            if value < 0:
                errors.append(f"{feature} cannot be negative.")
            values[feature] = value
        except ValueError:
            errors.append(f"Please enter a valid number for {feature}.")

    if errors:
        # Show the form again with error messages
        return render_template("index.html", show_form=True, errors=errors)

    # Arrange values in the correct order for the model
    input_array = np.array([[values[f] for f in FEATURES]])

    # Predict the price
    predicted_price = model.predict(input_array)[0]
    predicted_price = round(predicted_price, 2)

    # Format price in Indian numbering style (e.g., 12,34,567)
    formatted_price = format_indian_currency(predicted_price)

    return render_template(
        "result.html",
        predicted_price=formatted_price,
        inputs=values
    )


def format_indian_currency(amount):
    """
    Formats a number into the Indian numbering system.
    Example: 1234567 -> "12,34,567"
    """
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


if __name__ == "__main__":
    # debug=True gives helpful error messages during development.
    # Turn it off (debug=False) for a production deployment.
    app.run(debug=True)
