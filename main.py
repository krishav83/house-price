from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")  # Make sure model.pkl is in the same directory

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(request.form[key]) for key in request.form]
    final_features = np.array([features])
    prediction = model.predict(final_features)[0]
    formatted_prediction = f"${prediction:,.2f}"
    return render_template("result.html", prediction=formatted_prediction)

if __name__ == "__main__":
    app.run(debug=True)
