from flask import Flask, render_template, request
import joblib
import numpy as np

# Load model, scaler, features, and defaults
model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
features = joblib.load("model/features.pkl")
feature_means = joblib.load("model/feature_means.pkl")

# Select only important features for the form
important_features = ["alcohol", "sulphates", "citric acid", "volatile acidity"]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", features=important_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Fill defaults
        input_data = [feature_means[f] for f in features]

        # Replace with user values
        for i, feature in enumerate(features):
            if feature in important_features:
                input_data[i] = float(request.form[feature])

        # Prepare & scale
        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Predict
        prediction = model.predict(scaled_input)[0]
        result = f"🍷 Predicted Wine Quality: {prediction}"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
