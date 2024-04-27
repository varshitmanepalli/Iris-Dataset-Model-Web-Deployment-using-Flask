from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the Iris Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Expects data to be JSON with features in a list
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data['features'])])
    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
