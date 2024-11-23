import base64
import io
import os
import numpy as np
import mne
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from flask import Flask, request, jsonify, render_template, session, url_for
from flask_cors import CORS
from PIL import Image
import tempfile

app = Flask(__name__, static_folder='mindlock/static', template_folder='mindlock/templates')
app.secret_key = os.urandom(24)  # Add a secret key here
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
    
# Function to resize and compress the image and save it temporarily
def resize_image(fig, max_size=(500, 500)):
    # Save the figure to a temporary file in PNG format
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file_path = tmp_file.name
        fig.savefig(tmp_file_path, format='png', dpi=150)
    
    # Open image with PIL
    img = Image.open(tmp_file_path)

    # Resize image
    img.thumbnail(max_size)

    # Save resized image to a temporary buffer
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_resized_file:
        img.save(tmp_resized_file, format='PNG')
        tmp_resized_file_path = tmp_resized_file.name
    
    # Read the resized image into memory and convert it to Base64
    with open(tmp_resized_file_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Clean up temporary files
    try:
        os.remove(tmp_file_path)
        os.remove(tmp_resized_file_path)
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
    
    return img_base64
    
# Function to plot raw EEG data and return a compressed Base64 image
def plot_raw_data(raw, filename, events):
    fig = raw.plot(n_channels=64, scalings='auto', events=events, title=f"Raw EEG Data with Events: {filename}", show=False)
    img_base64 = resize_image(fig)
    return img_base64

# Function to plot epochs data and return a compressed Base64 image
def plot_epochs_data(epochs, event_id):
    fig = epochs.plot(n_channels=64, scalings='auto', events=epochs.events, event_id=event_id, title="Epochs with Event Markers", show=False)
    img_base64 = resize_image(fig)
    return img_base64

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

@app.route('/homepage', methods=['GET', 'POST'])
def homepage():
    username = session.get('username', 'Guest')
    plot_epochs_base64 = session.pop('plot_epochs_base64', None)

    return render_template(
        'homepage.html',
        username=username,
        plot_epochs_base64=plot_epochs_base64
    )


@app.route('/register', methods=['GET', 'POST'])
def register():
    # Your registration logic here
    return render_template('register.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    # Store the file in a variable (file object)
    stored_file = file  # You can now use 'stored_file' to refer to the uploaded file
    
    print(f"File uploaded: {stored_file.filename}")


    try:

        # Load the uploaded file directly into memory
        raw = mne.io.read_raw_fif(stored_file, preload=True)
        
        raw.set_channel_types({'Marker': 'stim'})
        montage = mne.channels.make_standard_montage('easycap-M1')
        raw.set_montage(montage)
        
        events = mne.find_events(raw, stim_channel= 'Marker')
        
        # Generate the plot of the raw EEG data
        plot_raw_base64 = plot_raw_data(raw, file.filename, events)
        
        raw.filter(l_freq=1.0, h_freq=30.0)
        
        unique_event_id = events[0, 2]
        event_id = {f"Event_{unique_event_id}": unique_event_id}
        
        tmin = -0.2
        tmax = 0.8
        baseline = (None, 0)
        
        epochs = mne.Epochs(raw, 
                            events = events,
                            event_id = event_id,
                            tmin = tmin,
                            tmax = tmax,
                            baseline = baseline,
                            preload = True)

        # Convert the data to a NumPy array (selecting the required channels)
        eeg_data = epochs.get_data()[:, :64, :]  # Select only the first 64 channels (adjust if necessary)
        print(f"EEG data shape for uploaded file: {eeg_data.shape}")
        
        # Generate the plot of the epochs data
        plot_epochs_base64 = plot_epochs_data(epochs, event_id)
        
        # Store the plot images in the session
        session['plot_raw_base64'] = plot_raw_base64
        session['plot_epochs_base64'] = plot_epochs_base64

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
            'message': 'Login successful'
        }, 200

    except Exception as e:
        print(f"Error in prediction for uploaded file: {str(e)}")
        return {'error': str(e)}, 500
    



if __name__ == '__main__':
    app.run(debug=True)
