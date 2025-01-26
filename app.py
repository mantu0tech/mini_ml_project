from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('iris_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        input_features = np.array(features).reshape(1, -1)
        prediction = loaded_model.predict(input_features)
        output = iris.target_names[prediction[0]]
    except Exception as e:
        return f"Error: {e}"

    return render_template('index.html', prediction_text=f'The predicted Iris species is: {output}')

if __name__ == '__main__':
    app.run(debug=True)
