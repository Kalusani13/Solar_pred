import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
import os

# create flask app
app = Flask(__name__)

model_path = os.path.join(os.getcwd(), "Dt_model.pkl")
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except (FileNotFoundError, IOError, EOFError, pickle.UnpicklingError):
    model = None

@app.route("/")
def Home():
    return render_template("Web_page.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("Web_page.html", prediction_text="Model failed to load.")

    # CHANGE 1: Get form values as a dictionary
    input_values = request.form.to_dict()

    float_features = [float(x) for x in input_values.values()]
    features = [np.array(float_features)]

    prediction = model.predict(features)

    # CHANGE 2: Pass input values to the template
    return render_template("Web_page.html", prediction_text=f"The solar Irradiation is {prediction[0]}", input_values=input_values)

if __name__ == "__main__":
    app.run()