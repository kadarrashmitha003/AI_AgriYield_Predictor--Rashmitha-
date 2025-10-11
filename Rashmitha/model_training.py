# AgriYield Predictor - Model Training
# This script trains the machine learning model and saves it as .pkl files

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🌱 AgriYield Predictor - Model Training")
print("=" * 50)

def train_and_save_model():
    # Step 1: Load the data
    print("\n📊 STEP 1: Loading data...")
    try:
        df = pd.read_csv(r"C:\Users\KADARUS\OneDrive\Desktop\infosys\raw data\Crop_recommendation.csv")
        print(f"✅ Data loaded successfully!")
        print(f"   Dataset shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("❌ Error: 'Crop_recommendation.csv' file not found!")
        print("   Please make sure the CSV file is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

    # Step 2: Explore the data
    print("\n🔍 STEP 2: Exploring data...")
    print(f"   Number of crops: {df['label'].nunique()}")
    print(f"   Crops available: {list(df['label'].unique())}")
    print(f"   Samples per crop: {df['label'].value_counts().iloc[0]} (balanced dataset)")

    # Show basic statistics
    print("\n📈 Basic statistics:")
    print(df.describe())

    # Step 3: Prepare features and target
    print("\n🛠️ STEP 3: Preparing data for machine learning...")
    
    # Separate features (X) and target (y)
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"   Features: {list(X.columns)}")
    print(f"   Target: Crop type")

    # Step 4: Encode the target labels (crop names to numbers)
    print("\n🔢 STEP 4: Encoding crop names...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("   Crop encoding mapping:")
    for i, crop in enumerate(label_encoder.classes_):
        print(f"     {crop} → {i}")

    # Step 5: Split the data into training and testing sets
    print("\n📊 STEP 5: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")

    # Step 6: Scale the features
    print("\n⚖️ STEP 6: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("   Features scaled using StandardScaler")

    # Step 7: Train the machine learning model
    print("\n🤖 STEP 7: Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        random_state=42,       # For reproducible results
        max_depth=10,          # Maximum depth of trees
        min_samples_split=5,   # Minimum samples required to split a node
        min_samples_leaf=2     # Minimum samples required at a leaf node
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    print("✅ Model trained successfully!")

    # Step 8: Evaluate the model
    print("\n🎯 STEP 8: Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Model Accuracy: {accuracy * 100:.2f}%")
    
    # Show classification report
    print("\n   Detailed performance report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Step 9: Feature importance
    print("\n📊 STEP 9: Analyzing feature importance...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("   Most important features for prediction:")
    for _, row in feature_importance.iterrows():
        print(f"     {row['feature']}: {row['importance']:.4f}")

    # Step 10: Save the model and encoders
    print("\n💾 STEP 10: Saving model and encoders...")
    try:
        # Save the trained model
        joblib.dump(model, 'crop_model.pkl')
        print("   ✅ crop_model.pkl saved")
        
        # Save the label encoder
        joblib.dump(label_encoder, 'label_encoder.pkl')
        print("   ✅ label_encoder.pkl saved")
        
        # Save the scaler
        joblib.dump(scaler, 'scaler.pkl')
        print("   ✅ scaler.pkl saved")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return False

    # Step 11: Test the saved model
    print("\n🧪 STEP 11: Testing saved model...")
    try:
        # Load the saved model and test
        loaded_model = joblib.load('crop_model.pkl')
        loaded_encoder = joblib.load('label_encoder.pkl')
        loaded_scaler = joblib.load('scaler.pkl')
        
        # Make a test prediction
        test_data = np.array([[90, 40, 40, 25, 80, 6.5, 200]])  # Sample input
        test_scaled = loaded_scaler.transform(test_data)
        test_pred = loaded_model.predict(test_scaled)
        test_crop = loaded_encoder.inverse_transform(test_pred)[0]
        
        print(f"   Test prediction successful!")
        print(f"   Input: [N=90, P=40, K=40, temp=25, humidity=80, pH=6.5, rainfall=200]")
        print(f"   Predicted crop: {test_crop}")
        
    except Exception as e:
        print(f"❌ Error testing saved model: {e}")
        return False

    # Final summary
    print("\n" + "=" * 50)
    print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📁 Files created:")
    print(f"   • crop_model.pkl - Trained machine learning model")
    print(f"   • label_encoder.pkl - Converts crop names to numbers")
    print(f"   • scaler.pkl - Scales input features")
    print(f"\n🌾 Model can predict: {len(label_encoder.classes_)} different crops")
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
    print(f"\n🚀 Next: Run 'python app.py' to start the web application")
    
    return True

def check_model_files():
    """Check if all required model files exist"""
    print("\n🔍 Checking for existing model files...")
    required_files = ['crop_model.pkl', 'label_encoder.pkl', 'scaler.pkl']
    all_exist = True
    
    for file in required_files:
        try:
            joblib.load(file)
            print(f"   ✅ {file} - Found")
        except:
            print(f"   ❌ {file} - Missing")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("AgriYield Predictor - Model Training System")
    print("This script will:")
    print("1. Load the crop recommendation dataset")
    print("2. Train a Random Forest machine learning model")
    print("3. Save the model as .pkl files for the web app")
    print("4. Test the model to ensure it works correctly")
    print("-" * 50)
    
    # Check if files already exist
    if check_model_files():
        response = input("\nModel files already exist. Retrain? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without retraining.")
            exit()
    
    # Train and save the model
    success = train_and_save_model()
    
    if success:
        print("\n✅ All files created successfully! You can now run the web application.")
    else:
        print("\n❌ Model training failed. Please check the error messages above.")