import os
import numpy as np
import mne
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tempfile
import scipy.io
import matplotlib
import pandas as pd
import shutil
from pathlib import Path
from sklearn.svm import SVC
from scipy.special import softmax

matplotlib.use('Agg')

app = Flask(__name__, static_folder='mindlock/static', template_folder='mindlock/templates')
CORS(app)  # Enable CORS

# Load pre-trained models (adjust the paths as necessary)
cnn_model_path = r'C:\Users\Vehuel\Downloads\Mindlock\for-train-model-1.keras'
svm_model_path = r'C:\Users\Vehuel\Downloads\Mindlock\for-train-svm_model.pkl'

try:
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    print("CNN model loaded successfully.")
except Exception as e:
    cnn_model = None
    print(f"Error loading CNN model: {e}")

try:
    svm_model = joblib.load(svm_model_path)
    print("SVM model loaded successfully.")
except Exception as e:
    svm_model = None
    print(f"Error loading SVM model: {e}")

def predict_eeg(file_path, subject_id):
    predictions = []
    try:
        # Load the .fif file with MNE
        raw = mne.read_epochs(file_path, preload=True)
        # Convert the data to a NumPy array (selecting the required channels, e.g., first 64)
        eeg_data = raw.get_data()[:64]  # Adjust according to your data
        # Check if the data is in the expected shape
        if eeg_data.ndim != 2:
            return {'error': f'Expected 2D EEG data for {file_path}, got shape {eeg_data.shape}'}, 400
        # Check if models are loaded
        if cnn_model is None or svm_model is None:
            return {'error': 'Model not loaded properly'}, 500
        # Pass the EEG data through the CNN to extract features
        cnn_features = cnn_model.predict(eeg_data)
        # Flatten the features for SVM input (to match the SVM training input shape)
        cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)
        # Predict using the SVM model
        svm_decision_scores = svm_model.decision_function(cnn_features)
        # Get the predicted class with the highest score
        predicted_class = np.argmax(svm_decision_scores)
        # Convert the predicted class index to the subject format (sub001, sub002, ...)
        predicted_subject = f"sub{predicted_class + 1:03d}"
        # Store the prediction
        predictions.append(predicted_subject)
        # Print the predicted subject for this file
        print(f"Predicted subject for {file_path}: {predicted_subject}")
        # Check if the predicted subject matches the entered subject ID
        if predicted_subject == subject_id:
            print(f"Username matches the predicted subject: {predicted_subject}")
            return {'success': True, 'message': 'Login successful', 'predicted_subject': predicted_subject}, 200
        else:
            print(f"Username does not match the predicted subject: {predicted_subject}")
            return {'success': False, 'message': 'Intruder detected', 'predicted_subject': predicted_subject}, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    subject_id = request.form.get('username')
    
    if not subject_id:
        return jsonify({'error': 'No username provided'}), 400
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        file.save(temp_file_path)
    
    result, status_code = predict_eeg(temp_file_path, subject_id)
    
    # Remove the temporary file after processing
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    
    return jsonify(result), status_code

if __name__ == '__main__':
    app.run(debug=True)
