import joblib

def predict_material(spectral_data, color_label):
    """
    Predicts the filament material given spectral data and a color label.
    
    Parameters:
        spectral_data (list): A list of 18 spectral channel values.
        color_label (str): A string representing the filament color ('R', 'B', 'G', etc.).
    
    Returns:
        str: Predicted filament material.
    """
    # Define paths
    model_dir = './models/'
    
    # Load the trained model and preprocessing tools
    scaler = joblib.load(model_dir + 'scaler.pkl')
    pca = joblib.load(model_dir + 'pca.pkl')
    model = joblib.load(model_dir + 'svm_model.pkl')
    material_encoder = joblib.load(model_dir + 'material_encoder.pkl')
    color_encoder = joblib.load(model_dir + 'color_encoder.pkl')
    
    # Validate input dimensions
    if len(spectral_data) != 18:
        raise ValueError("Spectral data must contain exactly 18 channel values.")
    
    # Encode the color
    encoded_color = color_encoder.transform([color_label])[0]
    
    # Combine spectral data with encoded color
    combined_sample = spectral_data + [encoded_color]
    
    # Scale the combined data
    scaled_sample = scaler.transform([combined_sample]).tolist()
    
    # Apply PCA
    pca_sample = pca.transform(scaled_sample).tolist()
    
    # Predict material type
    predicted_material_encoded = model.predict(pca_sample)
    predicted_material = material_encoder.inverse_transform(predicted_material_encoded)[0]
    
    return predicted_material
