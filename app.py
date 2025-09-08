from flask import Flask, request, render_template_string
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = 'knn_iris_model.pkl'

# Train and save model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    joblib.dump(knn, MODEL_PATH)

# Load model
model = joblib.load(MODEL_PATH)

# HTML template with embedded Jinja2 for form and output
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Iris KNN Prediction</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; }
      input[type=number] { width: 100px; padding: 5px; margin: 5px; }
      button { padding: 7px 15px; }
      .result { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Iris Species Prediction (KNN)</h1>
    <form method="POST">
        <label>Sepal Length (cm): <input type="number" step="0.1" name="sepal_length" required></label><br>
        <label>Sepal Width (cm): <input type="number" step="0.1" name="sepal_width" required></label><br>
        <label>Petal Length (cm): <input type="number" step="0.1" name="petal_length" required></label><br>
        <label>Petal Width (cm): <input type="number" step="0.1" name="petal_width" required></label><br><br>
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
    <div class="result">
        Prediction: <strong>{{ species_names[prediction] }}</strong> (Class: {{ prediction }})
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    species_names = load_iris().target_names

    if request.method == 'POST':
        try:
            # Parse input features from form
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
            features_np = np.array(features).reshape(1, -1)

            # Predict
            prediction = model.predict(features_np)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        species_names=species_names
    )


if __name__ == '__main__':
    app.run(debug=True)
