import os
import numpy as np
import mne
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from flask import Flask, request, jsonify, render_template, session, url_for
from flask_cors import CORS
import tempfile  # Import tempfile module

app = Flask(__name__, static_folder='mindlock/static', template_folder='mindlock/templates')
CORS(app)  # Enable CORS

# Load pre-trained models
cnn_model_path = r'C:\Users\Vehuel\Downloads\Mindlock\for-train-model-1.keras'
svm_model_path = r'C:\Users\Vehuel\Downloads\Mindlock\for-train-svm_model.pkl'

try:
    cnn_model = load_model(cnn_model_path)
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

def predict_eeg(file_path):
    try:
        # Load the .fif file with MNE
        raw = mne.read_epochs(file_path, preload=True)

        # Convert the data to a NumPy array (selecting the required channels)
        eeg_data = raw.get_data()[:, :64, :]  # Select only the first 64 channels (adjust if necessary)
        print(f"EEG data shape for {file_path}: {eeg_data.shape}")

        # Reshape data for CNN (1, num_channels, num_timepoints)
        if eeg_data.ndim == 3:
            eeg_data = eeg_data.reshape(eeg_data.shape[0], 64, -1)  # Reshape to (num_epochs, 64, num_timepoints)
        else:
            return {'error': f'Expected 3D EEG data for {file_path}, got shape {eeg_data.shape}'}, 400

        # Check if models are loaded
        if cnn_model is None or svm_model is None:
            return {'error': 'Model not loaded properly'}, 500

        # Pass the EEG data through the CNN to extract features
        cnn_features = cnn_model.predict(eeg_data)
        cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)

        # Predict using the SVM model
        svm_decision_scores = svm_model.decision_function(cnn_features)
        predicted_class = np.argmax(svm_decision_scores)
        predicted_subject = f"sub{predicted_class + 1:03d}"

        # Log the prediction in the desired format
        print(f"Predicted subject for {os.path.basename(file_path)}: {predicted_subject}")
        return {
            'success': True,
            'predicted_subject': predicted_subject,
            'message': f'Prediction successful for {file_path}'
        }, 200
    except Exception as e:
        print(f"Error in prediction for {file_path}: {str(e)}")
        return {'error': str(e)}, 500

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/homepage')
def homepage():
    # Make sure to handle session for username
    username = session.get('username', 'Guest')  # Default to 'Guest' if not logged in
    return render_template('homepage.html', username=username)

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Your registration logic here
    return render_template('register.html')  # or whatever your template is


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        # Load the uploaded file directly into memory
        raw = mne.read_epochs(file.stream, preload=True)

        # Convert the data to a NumPy array (selecting the required channels)
        eeg_data = raw.get_data()[:, :64, :]  # Select only the first 64 channels (adjust if necessary)
        print(f"EEG data shape for uploaded file: {eeg_data.shape}")

        # Reshape data for CNN (1, num_channels, num_timepoints)
        if eeg_data.ndim == 3:
            eeg_data = eeg_data.reshape(eeg_data.shape[0], 64, -1)  # Reshape to (num_epochs, 64, num_timepoints)
        else:
            return {'error': f'Expected 3D EEG data for uploaded file, got shape {eeg_data.shape}'}, 400

        # Check if models are loaded
        if cnn_model is None or svm_model is None:
            return {'error': 'Model not loaded properly'}, 500

        # Pass the EEG data through the CNN to extract features
        cnn_features = cnn_model.predict(eeg_data)
        cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)

        # Predict using the SVM model
        svm_decision_scores = svm_model.decision_function(cnn_features)
        predicted_class = np.argmax(svm_decision_scores)
        predicted_subject = f"sub{predicted_class + 1:03d}"

        # Print and return the predicted subject
        print(f"Predicted subject for {file.filename}: {predicted_subject}")
        return {
            'success': True,
            'predicted_subject': predicted_subject,
            'uploaded_filename': file.filename,
            'message': 'Login  successful'
        }, 200

    except Exception as e:
        print(f"Error in prediction for uploaded file: {str(e)}")
        return {'error': str(e)}, 500
    


if __name__ == '__main__':
    app.run(debug=True)
