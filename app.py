from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Function to convert experience text to numeric
def convert_experience(x):
    if isinstance(x, str):
        if x.isdigit():
            return int(x)
        else:
            word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                         'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                         'eleven': 11, 'twelve': 12, 'zero': 0}
            return word_dict.get(x.lower(), 0)
    return x

# Home route to render index.html
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    experience = data.get('experience', 0)
    test_score = float(data.get('test_score', 0))
    interview_score = float(data.get('interview_score', 0))
    
    # Convert experience to numeric
    experience = convert_experience(experience)
    
    # Prepare input for prediction
    input_features = np.array([[experience, test_score, interview_score]])
    
    # Make prediction
    prediction = model.predict(input_features)
    
    # Return prediction as JSON response
    return jsonify({'salary': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
