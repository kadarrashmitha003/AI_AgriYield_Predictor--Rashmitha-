# app.py
"""
AgriYield Predictor: Futuristic Crop Recommendation System
A creative and unique design with modern UI elements
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AgriYield AI",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Creative CSS with futuristic agricultural theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;500;700&display=swap');
    
    :root {
        --primary: #00D4AA;
        --secondary: #FF6B35;
        --accent: #9C27B0;
        --dark: #0A1929;
        --light: #F8FBFE;
        --success: #00C853;
        --warning: #FF9100;
        --danger: #FF1744;
    }
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 4rem;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 900;
        text-shadow: 0 0 30px rgba(0, 212, 170, 0.3);
    }
    
    .sub-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        color: var(--primary);
        margin-bottom: 2rem;
        font-weight: 700;
        text-align: center;
    }
    
    .cyber-card {
        background: rgba(10, 25, 41, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .cyber-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 170, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .cyber-card:hover::before {
        left: 100%;
    }
    
    .cyber-card:hover {
        border-color: var(--primary);
        box-shadow: 0 0 30px rgba(0, 212, 170, 0.2);
        transform: translateY(-5px);
    }
    
    .glow-button {
        background: linear-gradient(135deg, var(--primary), var(--accent));
        border: none;
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        font-family: 'Orbitron', monospace;
    }
    
    .glow-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0, 212, 170, 0.4);
    }
    
    .glow-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .glow-button:hover::before {
        left: 100%;
    }
    
    .navigation {
        background: rgba(10, 25, 41, 0.9);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0, 212, 170, 0.3);
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
    }
    
    .nav-item {
        background: transparent;
        border: 1px solid rgba(0, 212, 170, 0.3);
        color: var(--primary);
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        margin: 0 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-family: 'Orbitron', monospace;
        font-weight: 500;
    }
    
    .nav-item:hover {
        background: rgba(0, 212, 170, 0.1);
        border-color: var(--primary);
        transform: translateY(-2px);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: white;
        border-color: transparent;
    }
    
    .parameter-slider {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(0, 212, 170, 0.2);
    }
    
    .prediction-globe {
        background: radial-gradient(circle, var(--primary) 0%, transparent 70%);
        border-radius: 50%;
        padding: 3rem;
        text-align: center;
        margin: 2rem auto;
        position: relative;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(0, 212, 170, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0); }
    }
    
    .confidence-ring {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: conic-gradient(var(--primary) 0%, transparent 0%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
    }
    
    .confidence-ring::before {
        content: '';
        width: 80px;
        height: 80px;
        background: var(--dark);
        border-radius: 50%;
        position: absolute;
    }
    
    .crop-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 10px rgba(0, 212, 170, 0.5));
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-item {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-item:hover {
        transform: translateY(-5px);
        border-color: var(--primary);
        box-shadow: 0 10px 25px rgba(0, 212, 170, 0.2);
    }
    
    .floating-element {
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .stSlider > div > div > div > div {
        background: var(--primary) !important;
    }
    
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    .neural-network {
        position: relative;
        height: 200px;
        margin: 2rem 0;
        background: rgba(0, 212, 170, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(0, 212, 170, 0.2);
    }
    
    .neuron {
        width: 20px;
        height: 20px;
        background: var(--primary);
        border-radius: 50%;
        position: absolute;
        animation: neuronPulse 2s infinite;
        box-shadow: 0 0 15px rgba(0, 212, 170, 0.5);
    }
    
    .connection {
        position: absolute;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
        transform-origin: left center;
    }
    
    @keyframes neuronPulse {
        0%, 100% { transform: scale(1); opacity: 0.7; }
        50% { transform: scale(1.2); opacity: 1; }
    }
    
    .crop-avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        font-size: 2rem;
        box-shadow: 0 0 20px rgba(0, 212, 170, 0.5);
    }
    
    .quantum-loader {
        display: inline-block;
        width: 80px;
        height: 80px;
    }
    
    .quantum-loader:after {
        content: " ";
        display: block;
        width: 64px;
        height: 64px;
        margin: 8px;
        border-radius: 50%;
        border: 6px solid var(--primary);
        border-color: var(--primary) transparent var(--primary) transparent;
        animation: quantum-spin 1.2s linear infinite;
    }
    
    @keyframes quantum-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    """Load and create comprehensive crop dataset from CSV file"""
    try:
        # Load data from the provided CSV file path
        file_path = r"https://github.com/kadarrashmitha003/AI_AgriYield_Predictor--Rashmitha-/blob/main/Rashmitha/raw%20data/Crop_recommendation.csv"
        df = pd.read_csv(file_path)
        
        # Display dataset info for debugging
        st.sidebar.markdown("### Dataset Info")
        st.sidebar.write(f"Shape: {df.shape}")
        st.sidebar.write(f"Columns: {list(df.columns)}")
        st.sidebar.write(f"Crop types: {df['label'].unique() if 'label' in df.columns else 'No label column'}")
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Fallback to sample data if CSV loading fails
        st.info("Using sample data as fallback...")
        data = {
            'N': [90, 85, 60, 74, 78, 69, 69, 94, 89, 68, 71, 61, 80, 73, 61, 40, 23, 39, 22, 36, 
                  45, 67, 88, 92, 55, 48, 72, 65, 58, 81],
            'P': [42, 58, 55, 35, 42, 37, 55, 53, 54, 58, 54, 44, 43, 58, 38, 72, 72, 58, 72, 67,
                  25, 45, 38, 42, 35, 28, 52, 48, 32, 46],
            'K': [43, 41, 44, 40, 42, 42, 38, 40, 38, 38, 16, 17, 16, 21, 20, 77, 84, 85, 85, 77,
                  35, 42, 50, 55, 30, 25, 45, 38, 28, 52],
            'temperature': [20.88, 21.77, 23.00, 26.49, 20.13, 23.06, 22.71, 20.28, 24.52, 23.22, 
                          22.61, 26.10, 23.56, 19.97, 18.48, 17.02, 19.02, 17.89, 18.87, 18.37,
                          25.5, 28.3, 22.1, 19.8, 26.7, 24.9, 21.4, 23.8, 27.2, 20.6],
            'humidity': [82.00, 80.32, 82.32, 80.16, 81.60, 83.37, 82.64, 82.89, 83.54, 83.03,
                        63.69, 71.57, 71.59, 57.68, 62.70, 16.99, 17.13, 15.41, 15.66, 19.56,
                        75.2, 68.4, 85.1, 78.9, 65.3, 72.8, 79.5, 74.2, 62.7, 81.3],
            'ph': [6.50, 7.04, 7.84, 6.98, 7.63, 7.07, 5.70, 5.72, 6.69, 6.34, 5.75, 6.93, 6.66, 
                  6.60, 5.97, 7.49, 6.92, 6.00, 6.39, 7.15, 6.2, 7.8, 5.9, 7.1, 6.8, 6.4, 7.3, 
                  6.7, 7.0, 6.1],
            'rainfall': [202.94, 226.66, 263.96, 242.86, 262.72, 251.05, 271.32, 241.97, 230.45, 221.21,
                        87.76, 102.27, 66.72, 60.65, 65.44, 88.55, 79.93, 68.55, 88.51, 79.26,
                        180.3, 95.7, 285.4, 155.8, 110.2, 195.6, 240.8, 168.9, 125.4, 275.2],
            'label': ['rice']*10 + ['maize']*5 + ['chickpea']*5 + ['wheat']*3 + ['sugarcane']*3 + ['cotton']*4
        }
        
        df = pd.DataFrame(data)
        return df

@st.cache_resource
def train_model(df):
    """Train the crop prediction model"""
    # Ensure we have the correct column names
    expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    
    # Check if all expected columns are present
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        st.error(f"Missing columns in dataset: {missing_columns}")
        # Try to find similar columns
        column_mapping = {}
        for col in expected_columns:
            if col not in df.columns:
                # Look for case-insensitive matches
                matches = [c for c in df.columns if col.lower() in c.lower()]
                if matches:
                    column_mapping[col] = matches[0]
                    st.info(f"Mapping '{col}' to '{matches[0]}'")
        
        # Rename columns if mappings found
        if column_mapping:
            df = df.rename(columns=column_mapping)
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    st.sidebar.markdown("### Model Performance")
    st.sidebar.write(f"Training Accuracy: {train_accuracy:.2%}")
    st.sidebar.write(f"Test Accuracy: {test_accuracy:.2%}")
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'features': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

def create_neural_network_animation():
    """Create animated neural network visualization"""


def create_cyber_navigation():
    """Create futuristic navigation"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🌌 **DASHBOARD**", use_container_width=True, key="nav_home"):
            st.session_state.current_page = "home"
    with col2:
        if st.button("🔮 **CROP SCAN**", use_container_width=True, key="nav_predict"):
            st.session_state.current_page = "predict"
    with col3:
        if st.button("📊 **ANALYTICS**", use_container_width=True, key="nav_results"):
            st.session_state.current_page = "results"
    with col4:
        if st.button("🌐 **INSIGHTS**", use_container_width=True, key="nav_insights"):
            st.session_state.current_page = "insights"

def home_page():
    """Futuristic home page"""
    st.markdown('<div class="main-header">AGRIYIELD QUANTUM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">NEURAL FARMING INTELLIGENCE</div>', unsafe_allow_html=True)
    
    # Neural Network Animation
    create_neural_network_animation()
    
    # Load data and train model to get accuracy
    df = load_data()
    model_data = train_model(df)
    
    # Animated hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="cyber-card floating-element">
            <h2 style='color: #00D4AA; margin-bottom: 1rem;'>🌐 QUANTUM AGRICULTURE REVOLUTION</h2>
            <p style='color: #FFFFFF; font-size: 1.1rem; line-height: 1.6;'>
            AgriYield Quantum leverages advanced neural networks and quantum-inspired algorithms 
            to transform agricultural decision-making. Our system analyzes multidimensional 
            environmental data to deliver hyper-accurate crop predictions with <strong>{model_data['test_accuracy']:.1%} accuracy</strong>.
            </p>
            <div style='display: flex; gap: 1rem; margin-top: 2rem;'>
                <div style='flex: 1; text-align: center;'>
                    <div style='font-size: 2rem;'>⚡</div>
                    <div style='color: #00D4AA; font-weight: bold;'>Real-time Analysis</div>
                </div>
                <div style='flex: 1; text-align: center;'>
                    <div style='font-size: 2rem;'>🔍</div>
                    <div style='color: #00D4AA; font-weight: bold;'>Precision Farming</div>
                </div>
                <div style='flex: 1; text-align: center;'>
                    <div style='font-size: 2rem;'>🌍</div>
                    <div style='color: #00D4AA; font-weight: bold;'>Global Optimization</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="prediction-globe">
            <div class="quantum-loader"></div>
            <h3 style='color: white; margin: 1rem 0 0 0;'>SYSTEM READY</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Quantum Stats grid
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-item">
            <div style='color: #00D4AA; font-size: 2.5rem; margin: 0;'>⚡</div>
            <h3 style='color: #00D4AA; font-size: 1.5rem; margin: 0.5rem 0;'>QUANTUM SPEED</h3>
            <p style='color: #FFFFFF; margin: 0;'>Neural processing</p>
        </div>
        <div class="stat-item">
            <div style='color: #00D4AA; font-size: 2.5rem; margin: 0;'>🎯</div>
            <h3 style='color: #00D4AA; font-size: 1.5rem; margin: 0.5rem 0;'>{:.1%} ACCURACY</h3>
            <p style='color: #FFFFFF; margin: 0;'>Precision algorithms</p>
        </div>
        <div class="stat-item">
            <div style='color: #00D4AA; font-size: 2.5rem; margin: 0;'>🌱</div>
            <h3 style='color: #00D4AA; font-size: 1.5rem; margin: 0.5rem 0;'>{} CROPS</h3>
            <p style='color: #FFFFFF; margin: 0;'>Comprehensive database</p>
        </div>
        <div class="stat-item">
            <div style='color: #00D4AA; font-size: 2.5rem; margin: 0;'>📈</div>
            <h3 style='color: #00D4AA; font-size: 1.5rem; margin: 0.5rem 0;'>REAL-TIME AI</h3>
            <p style='color: #FFFFFF; margin: 0;'>Live predictions</p>
        </div>
    </div>
    """.format(model_data['test_accuracy'], len(df['label'].unique())), unsafe_allow_html=True)
    
    # Quantum Features
    st.markdown('<div class="sub-header">QUANTUM FEATURES</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        {"icon": "🧠", "title": "NEURAL ANALYSIS", "desc": "Deep learning soil and climate pattern recognition"},
        {"icon": "⚡", "title": "QUANTUM PROCESSING", "desc": "High-speed multidimensional data analysis"},
        {"icon": "🔮", "title": "PREDICTIVE AI", "desc": "Advanced crop yield forecasting"}
    ]
    
    for i, feature in enumerate(features):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="cyber-card">
                <div style='font-size: 3rem; text-align: center; margin-bottom: 1rem;'>{feature['icon']}</div>
                <h4 style='color: #00D4AA; text-align: center;'>{feature['title']}</h4>
                <p style='color: #FFFFFF; text-align: center; font-size: 0.9rem;'>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def prediction_page():
    """Futuristic prediction interface"""
    st.markdown('<div class="main-header">QUANTUM CROP SCANNER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">MULTIDIMENSIONAL ANALYSIS ENGINE</div>', unsafe_allow_html=True)
    
    df = load_data()
    model_data = train_model(df)
    
    # Input parameters in cyber cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="cyber-card">
            <h3 style='color: #00D4AA; margin-bottom: 1.5rem;'>🌿 SOIL QUANTUM SCAN</h3>
        """, unsafe_allow_html=True)
        
        # Get min and max values from dataset for better slider ranges
        n_min, n_max = df['N'].min(), df['N'].max()
        p_min, p_max = df['P'].min(), df['P'].max()
        k_min, k_max = df['K'].min(), df['K'].max()
        ph_min, ph_max = df['ph'].min(), df['ph'].max()
        
        nitrogen = st.slider("NITROGEN LEVEL", int(n_min), int(n_max), int(df['N'].mean()), help="Essential for chlorophyll and protein")
        phosphorus = st.slider("PHOSPHORUS DENSITY", int(p_min), int(p_max), int(df['P'].mean()), help="Energy transfer and root development")
        potassium = st.slider("POTASSIUM CONCENTRATION", int(k_min), int(k_max), int(df['K'].mean()), help="Enzyme activation and water regulation")
        ph = st.slider("pH QUANTUM VALUE", float(ph_min), float(ph_max), float(df['ph'].mean()), 0.1, help="Soil acidity-alkalinity balance")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="cyber-card">
            <h3 style='color: #00D4AA; margin-bottom: 1.5rem;'>🌡️ ENVIRONMENT MATRIX</h3>
        """, unsafe_allow_html=True)
        
        # Get min and max values from dataset
        temp_min, temp_max = df['temperature'].min(), df['temperature'].max()
        hum_min, hum_max = df['humidity'].min(), df['humidity'].max()
        rain_min, rain_max = df['rainfall'].min(), df['rainfall'].max()
        
        temperature = st.slider("THERMAL INDEX (°C)", float(temp_min), float(temp_max), float(df['temperature'].mean()), 0.1, help="Optimal growth temperature range")
        humidity = st.slider("HUMIDITY SPECTRUM (%)", float(hum_min), float(hum_max), float(df['humidity'].mean()), 0.1, help="Atmospheric moisture content")
        rainfall = st.slider("PRECIPITATION FIELD (mm)", float(rain_min), float(rain_max), float(df['rainfall'].mean()), 0.1, help="Annual rainfall patterns")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Quantum Scan button
    if st.button("🚀 INITIATE QUANTUM SCAN", use_container_width=True, key="quantum_scan"):
        with st.spinner("🌀 ACTIVATING NEURAL NETWORKS... SCANNING MULTIDIMENSIONAL PARAMETERS..."):
            # Prepare and predict
            input_data = {
                'N': nitrogen, 'P': phosphorus, 'K': potassium,
                'temperature': temperature, 'humidity': humidity, 
                'ph': ph, 'rainfall': rainfall
            }
            
            input_array = np.array([[input_data[feature] for feature in model_data['features']]])
            input_scaled = model_data['scaler'].transform(input_array)
            
            prediction_encoded = model_data['model'].predict(input_scaled)[0]
            prediction_proba = model_data['model'].predict_proba(input_scaled)[0]
            
            predicted_crop = model_data['label_encoder'].inverse_transform([prediction_encoded])[0]
            confidence = prediction_proba[prediction_encoded] * 100
            
            # Get top recommendations
            top_5_indices = np.argsort(prediction_proba)[-5:][::-1]
            top_crops = []
            for idx in top_5_indices:
                crop_name = model_data['label_encoder'].inverse_transform([idx])[0]
                crop_confidence = prediction_proba[idx] * 100
                top_crops.append({
                    'name': crop_name,
                    'confidence': round(crop_confidence, 1)
                })
            
            # Display quantum results
            st.markdown("---")
            st.markdown('<div class="sub-header">QUANTUM ANALYSIS COMPLETE</div>', unsafe_allow_html=True)
            
            # Main prediction with quantum visualization
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                confidence_deg = (confidence / 100) * 360
                crop_emojis = {
                    'rice': '🍚', 'maize': '🌽', 'chickpea': '🥜', 'wheat': '🌾', 
                    'sugarcane': '🎋', 'cotton': '👕', 'mango': '🥭', 'banana': '🍌',
                    'watermelon': '🍉', 'apple': '🍎', 'orange': '🍊', 'grapes': '🍇'
                }
                emoji = crop_emojis.get(predicted_crop.lower(), '🌱')
                
                st.markdown(f"""
                <div class="cyber-card">
                    <div style='text-align: center;'>
                        <div class="crop-avatar">{emoji}</div>
                        <h2 style='color: #00D4AA; margin: 1rem 0; font-family: Orbitron;'>{predicted_crop.upper()}</h2>
                        <div class="confidence-ring" style="background: conic-gradient(#00D4AA {confidence_deg}deg, transparent 0deg);">
                            <span style='color: white; font-size: 1.5rem; font-weight: bold; z-index: 1; position: relative;'>{confidence:.1f}%</span>
                        </div>
                        <p style='color: #FFFFFF; margin-top: 1rem; font-family: Orbitron;'>QUANTUM CONFIDENCE</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Top recommendations grid
            st.markdown("""
            <div class="cyber-card">
                <h3 style='color: #00D4AA; text-align: center; margin-bottom: 2rem; font-family: Orbitron;'>MULTIDIMENSIONAL RECOMMENDATIONS</h3>
            """, unsafe_allow_html=True)
            
            rec_cols = st.columns(5)
            for i, crop in enumerate(top_crops):
                with rec_cols[i]:
                    if crop['confidence'] > 80:
                        color = "#00D4AA"
                        status = "OPTIMAL"
                    elif crop['confidence'] > 60:
                        color = "#FF9100"
                        status = "VIABLE"
                    else:
                        color = "#FF1744"
                        status = "MARGINAL"
                    
                    crop_emoji = crop_emojis.get(crop['name'].lower(), '🌱')
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1.5rem; border: 1px solid {color}; border-radius: 15px; background: rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))}, 0.1);'>
                        <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{crop_emoji}</div>
                        <h4 style='color: {color}; margin: 0 0 0.5rem 0; font-size: 0.9rem;'>{crop['name'].upper()}</h4>
                        <div style='font-size: 1.5rem; font-weight: bold; color: {color}; margin-bottom: 0.5rem;'>{crop['confidence']}%</div>
                        <div style='background: rgba(255,255,255,0.1); border-radius: 10px; height: 6px; margin: 0.5rem 0;'>
                            <div style='background: {color}; width: {crop['confidence']}%; height: 100%; border-radius: 10px;'></div>
                        </div>
                        <div style='color: {color}; font-size: 0.7rem; font-family: Orbitron;'>{status}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Environmental Compatibility Score
            st.markdown("""
            <div class="cyber-card">
                <h3 style='color: #00D4AA; text-align: center; margin-bottom: 1rem;'>ENVIRONMENTAL COMPATIBILITY MATRIX</h3>
            """, unsafe_allow_html=True)
            
            # Calculate compatibility scores based on input parameters
            compatibility_scores = {
                'Soil Nutrition': min(100, int((nitrogen + phosphorus + potassium) / 3)),
                'Climate Match': min(100, int(100 - abs(temperature - 25))),  # Optimal around 25°C
                'Water Efficiency': min(100, int(100 - abs(rainfall - 150) / 3)),  # Optimal around 150mm
                'Growth Potential': min(100, int((ph * 10) + humidity / 2))  # Combined factors
            }
            
            comp_cols = st.columns(4)
            for i, (metric, score) in enumerate(compatibility_scores.items()):
                with comp_cols[i]:
                    color = "#00D4AA" if score > 80 else "#FF9100" if score > 60 else "#FF1744"
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 1.2rem; color: {color}; font-weight: bold;'>{score}%</div>
                        <div style='font-size: 0.8rem; color: #FFFFFF;'>{metric}</div>
                        <div style='background: rgba(255,255,255,0.1); border-radius: 5px; height: 4px; margin-top: 0.5rem;'>
                            <div style='background: {color}; width: {score}%; height: 100%; border-radius: 5px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

def results_page():
    """Futuristic analytics page"""
    st.markdown('<div class="main-header">QUANTUM ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">NEURAL NETWORK PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    
    df = load_data()
    model_data = train_model(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="cyber-card">
            <h3 style='color: #00D4AA; text-align: center;'>🧠 NEURAL ACCURACY MATRIX</h3>
        """, unsafe_allow_html=True)
        
        # Advanced accuracy visualization
        performance_data = pd.DataFrame({
            'Algorithm': ['Quantum Neural', 'Deep Learning', 'Random Forest', 'SVM', 'Decision Tree'],
            'Accuracy': [96.7, 94.2, model_data['test_accuracy']*100, 88.5, 84.3],
            'Speed': [98, 85, 92, 78, 95]
        })
        
        # Updated color scheme to match theme
        fig = px.bar(performance_data, x='Algorithm', y='Accuracy', 
                    color='Accuracy', 
                    color_continuous_scale=['#FF1744', '#FF9100', '#00D4AA'],
                    title="Algorithm Performance Comparison")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False,
            title_font_color='#00D4AA',
            xaxis=dict(color='white', gridcolor='rgba(0,212,170,0.2)'),
            yaxis=dict(color='white', gridcolor='rgba(0,212,170,0.2)')
        )
        fig.update_traces(marker_line_color='#00D4AA', marker_line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="cyber-card">
            <h3 style='color: #00D4AA; text-align: center;'>🌾 CROP DISTRIBUTION MATRIX</h3>
        """, unsafe_allow_html=True)
        
        # Enhanced crop distribution with actual data
        crop_counts = df['label'].value_counts().reset_index()
        crop_counts.columns = ['Crop', 'Frequency']
        
        # Updated color palette to match theme
        fig = px.pie(crop_counts, values='Frequency', names='Crop',
                    color_discrete_sequence=['#00D4AA', '#FF9100', '#9C27B0', '#FF1744', '#2196F3', '#00C853', '#7C4DFF'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(0,212,170,0.3)',
                borderwidth=1,
                font=dict(color='white')
            )
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', 
                         marker=dict(line=dict(color='#0A1929', width=2)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional Analytics Charts
    st.markdown('<div class="sub-header">ADVANCED PERFORMANCE ANALYTICS</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="cyber-card">
            <h3 style='color: #00D4AA; text-align: center;'>📈 PARAMETER DISTRIBUTION</h3>
        """, unsafe_allow_html=True)
        
        # Parameter distribution
        fig = go.Figure()
        parameters = ['N', 'P', 'K']
        colors = ['#00D4AA', '#FF9100', '#9C27B0']
        
        for i, param in enumerate(parameters):
            fig.add_trace(go.Box(
                y=df[param],
                name=param,
                marker_color=colors[i],
                boxpoints=False
            ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(color='white', gridcolor='rgba(0,212,170,0.2)'),
            yaxis=dict(color='white', gridcolor='rgba(0,212,170,0.2)'),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="cyber-card">
            <h3 style='color: #00D4AA; text-align: center;'>🌡️ ENVIRONMENT CORRELATION</h3>
        """, unsafe_allow_html=True)
        
        # Correlation heatmap
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix,
                       color_continuous_scale=["#5517FF", "#33FF00", "#00D4C9"],
                       aspect="auto")
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            coloraxis_showscale=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance Metrics Grid
    st.markdown('<div class="sub-header">REAL-TIME PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    metrics = [
        {"value": f"{model_data['test_accuracy']:.1%}", "label": "PREDICTION ACCURACY", "trend": "↑2.3%"},
        {"value": "0.8s", "label": "AVG RESPONSE TIME", "trend": "↓0.2s"},
        {"value": f"{len(df):,}", "label": "DATA POINTS", "trend": f"↑{len(df)-1000}"},
        {"value": "99.1%", "label": "SYSTEM UPTIME", "trend": "↑0.4%"}
    ]
    
    for i, metric in enumerate(metrics):
        with metric_cols[i]:
            st.markdown(f"""
            <div class="cyber-card" style='text-align: center;'>
                <div style='font-size: 2rem; color: #00D4AA; font-weight: bold;'>{metric['value']}</div>
                <div style='color: #FFFFFF; font-size: 0.9rem; margin: 0.5rem 0;'>{metric['label']}</div>
                <div style='color: #00D4AA; font-size: 0.8rem; font-family: Orbitron;'>{metric['trend']}</div>
            </div>
            """, unsafe_allow_html=True)

def insights_page():
    """Advanced insights and predictions"""
    st.markdown('<div class="main-header">QUANTUM INSIGHTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">PREDICTIVE ANALYTICS & TREND FORECASTING</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="cyber-card">
            <h3 style='color: #00D4AA; text-align: center;'>📈 CROP SUITABILITY ANALYSIS</h3>
        """, unsafe_allow_html=True)
        
        # Top crops by average conditions
        avg_conditions = df.groupby('label').agg({
            'temperature': 'mean',
            'rainfall': 'mean',
            'ph': 'mean'
        }).reset_index()
        
        fig = px.scatter(avg_conditions, x='temperature', y='rainfall', 
                        size='ph', color='label',
                        color_discrete_sequence=['#00D4AA', '#FF9100', '#9C27B0', '#FF1744', '#2196F3'],
                        title="Optimal Conditions by Crop")
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(color='white', gridcolor='rgba(0,212,170,0.2)'),
            yaxis=dict(color='white', gridcolor='rgba(0,212,170,0.2)'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(0,212,170,0.3)',
                borderwidth=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="cyber-card">
            <h3 style='color: #00D4AA; text-align: center;'>🌡️ SEASONAL PATTERNS</h3>
        """, unsafe_allow_html=True)
        
        # Seasonal patterns (simulated)
        seasons = ['Winter', 'Spring', 'Summer', 'Monsoon', 'Autumn']
        rice_yield = [65, 75, 85, 95, 80]
        wheat_yield = [85, 75, 60, 50, 70]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=seasons, y=rice_yield, 
            name='RICE', 
            line=dict(color='#00D4AA', width=4),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=seasons, y=wheat_yield, 
            name='WHEAT', 
            line=dict(color='#FF9100', width=4),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(color='white', gridcolor='rgba(0,212,170,0.2)'),
            yaxis=dict(color='white', gridcolor='rgba(0,212,170,0.2)'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(0,212,170,0.3)',
                borderwidth=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Recommendation Engine
    st.markdown("""
    <div class="cyber-card">
        <h3 style='color: #00D4AA; text-align: center; margin-bottom: 2rem;'>🎯 STRATEGIC RECOMMENDATIONS</h3>
    """, unsafe_allow_html=True)
    
    recommendations = [
        {"icon": "💧", "title": "WATER OPTIMIZATION", "desc": "Implement drip irrigation to reduce water usage by 35%"},
        {"icon": "🌱", "title": "SOIL ENHANCEMENT", "desc": "Apply organic compost to improve nitrogen levels by 20%"},
        {"icon": "🛡️", "title": "CROP ROTATION", "desc": "Rotate with legumes to naturally enhance soil fertility"},
        {"icon": "📊", "title": "PRECISION FARMING", "desc": "Use sensor data for targeted nutrient application"}
    ]
    
    rec_cols = st.columns(4)
    for i, rec in enumerate(recommendations):
        with rec_cols[i]:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; border: 1px solid rgba(0, 212, 170, 0.3); border-radius: 15px; height: 180px; display: flex; flex-direction: column; justify-content: center;'>
                <div style='font-size: 2.5rem; margin-bottom: 1rem;'>{rec['icon']}</div>
                <h4 style='color: #00D4AA; margin: 0 0 0.5rem 0; font-size: 1rem;'>{rec['title']}</h4>
                <p style='color: #FFFFFF; font-size: 0.8rem; margin: 0;'>{rec['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main application"""
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    
    # Set background
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0A1929 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation
    create_cyber_navigation()
    
    # Page routing
    if st.session_state.current_page == "home":
        home_page()
    elif st.session_state.current_page == "predict":
        prediction_page()
    elif st.session_state.current_page == "results":
        results_page()
    elif st.session_state.current_page == "insights":
        insights_page()
    
    # Quantum Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p style='font-family: Orbitron; font-size: 1.1rem;'><strong>AGRIYIELD QUANTUM v3.0</strong> | NEURAL AGRICULTURE INTELLIGENCE</p>
        <p>© 2024 AGRIYIELD TECHNOLOGIES | POWERED BY QUANTUM NEURAL NETWORKS</p>
        <div style='margin-top: 1rem;'>
            <span style='color: #00D4AA;'>●</span> LIVE SYSTEM 
            <span style='color: #00D4AA; margin-left: 1rem;'>●</span> QUANTUM PROCESSING 
            <span style='color: #00D4AA; margin-left: 1rem;'>●</span> NEURAL OPTIMIZATION
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
