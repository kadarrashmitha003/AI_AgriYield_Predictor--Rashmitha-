from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and encoders
try:
    model = joblib.load('crop_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Models loaded successfully!")
except:
    print("❌ Error loading models. Please run model_training.py first.")
    model = None
    label_encoder = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    try:
        # Get form data
        data = {
            'N': float(request.form['nitrogen']),
            'P': float(request.form['phosphorus']),
            'K': float(request.form['potassium']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall'])
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get crop name and confidence
        predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = prediction_proba[prediction_encoded] * 100
        
        # Get top 3 recommendations
        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
        top_crops = []
        for idx in top_3_indices:
            crop_name = label_encoder.inverse_transform([idx])[0]
            crop_confidence = prediction_proba[idx] * 100
            top_crops.append({
                'name': crop_name,
                'confidence': round(crop_confidence, 1)
            })
        
        # Get ideal conditions for predicted crop
        crop_data = pd.read_csv(r"C:\Users\KADARUS\OneDrive\Desktop\infosys\raw data\Crop_recommendation.csv")
        ideal_conditions = crop_data[crop_data['label'] == predicted_crop].describe().loc[['mean', 'min', 'max']]
        
        result = {
            'success': True,
            'predicted_crop': predicted_crop,
            'confidence': round(confidence, 1),
            'top_recommendations': top_crops,
            'input_data': data,
            'ideal_conditions': {
                'temperature': {
                    'mean': round(ideal_conditions['temperature']['mean'], 1),
                    'min': round(ideal_conditions['temperature']['min'], 1),
                    'max': round(ideal_conditions['temperature']['max'], 1)
                },
                'rainfall': {
                    'mean': round(ideal_conditions['rainfall']['mean'], 1),
                    'min': round(ideal_conditions['rainfall']['min'], 1),
                    'max': round(ideal_conditions['rainfall']['max'], 1)
                },
                'ph': {
                    'mean': round(ideal_conditions['ph']['mean'], 2),
                    'min': round(ideal_conditions['ph']['min'], 2),
                    'max': round(ideal_conditions['ph']['max'], 2)
                }
            }
        }
        
        return render_template('results.html', result=result)
        
    except Exception as e:
        return render_template('results.html', 
                            result={'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        input_data = {
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'ph': float(data['ph']),
            'rainfall': float(data['rainfall'])
        }
        
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = prediction_proba[prediction_encoded] * 100
        
        return jsonify({
            'predicted_crop': predicted_crop,
            'confidence': round(confidence, 1),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)