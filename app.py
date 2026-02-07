"""
Flask Web Application for Fake AI Image Detection
Author: Siddhant Mishra
University: Sage University Indore
Course: MCA Final Year Project

This web app allows users to upload images and get predictions.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)
CORS(app)

# Global variables
MODEL = None
MODEL_PATH = "results/best_model.keras"
IMG_SIZE = 224
CLASS_NAMES = ['Real Image', 'AI-Generated/Fake Image']

# Performance disclaimer
DISCLAIMER = """
⚠️ IMPORTANT: This model was trained on a specific dataset containing real photographs 
and AI-generated images from Stable Diffusion and Midjourney. While it achieves 90-95% 
accuracy on the test dataset, performance may vary on:
• Images from other AI generators (DALL-E, Firefly, etc.)
• Different image domains or styles
• Heavily edited or compressed images
• Images from sources not represented in training data

This is a research/academic project and should not be used as the sole method for 
authenticating images in critical applications.
"""


def load_model():
    """Load the trained model"""
    global MODEL
    
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print("Please train the model first using train_model.py")
        return None
    
    try:
        MODEL = keras.models.load_model(str(model_path))
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
        return MODEL
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def preprocess_image(image_bytes):
    """
    Preprocess uploaded image for prediction
    
    Args:
        image_bytes: Image bytes from upload
    
    Returns:
        Preprocessed image array
    """
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Resize
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def get_prediction(image_bytes):
    """
    Get prediction for uploaded image
    
    Args:
        image_bytes: Image bytes from upload
    
    Returns:
        Dictionary with prediction results
    """
    if MODEL is None:
        return {
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }
    
    try:
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Get prediction
        predictions = MODEL.predict(img_array, verbose=0)
        
        # Get probabilities
        real_prob = float(predictions[0][0])
        fake_prob = float(predictions[0][1])
        
        # Determine class
        predicted_class = 0 if real_prob > fake_prob else 1
        confidence = max(real_prob, fake_prob)
        
        # Confidence level description
        if confidence >= 0.9:
            conf_level = "Very High"
        elif confidence >= 0.75:
            conf_level = "High"
        elif confidence >= 0.6:
            conf_level = "Moderate"
        else:
            conf_level = "Low"
        
        return {
            'success': True,
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': confidence * 100,
            'confidence_level': conf_level,
            'probabilities': {
                'real': real_prob * 100,
                'fake': fake_prob * 100
            },
            'disclaimer': DISCLAIMER
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', disclaimer=DISCLAIMER)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'success': False})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'success': False})
    
    try:
        # Read file bytes
        image_bytes = file.read()
        
        # Get prediction
        result = get_prediction(image_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction request"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded', 'success': False})
    
    files = request.files.getlist('files[]')
    
    if len(files) == 0:
        return jsonify({'error': 'No files selected', 'success': False})
    
    results = []
    
    for file in files:
        try:
            image_bytes = file.read()
            result = get_prediction(image_bytes)
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'success': False
            })
    
    return jsonify({
        'success': True,
        'results': results,
        'total': len(results)
    })


@app.route('/model_info')
def model_info():
    """Get model information"""
    if MODEL is None:
        return jsonify({
            'loaded': False,
            'message': 'Model not loaded'
        })
    
    # Try to load results from training
    results_file = Path('results/test_results.json')
    config_file = Path('results/config.json')
    
    info = {
        'loaded': True,
        'input_shape': [IMG_SIZE, IMG_SIZE, 3],
        'classes': CLASS_NAMES,
        'total_parameters': MODEL.count_params()
    }
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            test_results = json.load(f)
            info['test_accuracy'] = f"{test_results['test_accuracy']*100:.2f}%"
            info['test_f1_score'] = f"{test_results['test_f1_score']*100:.2f}%"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            info['architecture'] = config.get('architecture', 'Unknown')
    
    return jsonify(info)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("FAKE AI IMAGE DETECTION - WEB APPLICATION")
    print("University: Sage University Indore")
    print("="*70)
    
    # Load model
    load_model()
    
    if MODEL is not None:
        print("\n🌐 Starting web server...")
        print(f"📍 Access the app at: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        print("="*70 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Cannot start server: Model not loaded")
        print("Please train the model first by running:")
        print("  python train_model.py")
