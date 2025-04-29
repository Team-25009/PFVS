import os
import joblib
import numpy as np

# Standalone material-prediction function using saved pipeline in ./models_per_color
def predict_material(spectral_array, color_label):
    """
    Predicts filament material from a spectral scan and color label.

    Assumes the current working directory contains a 'models_per_color' folder with:
      - material_encoder.pkl
      - <Color>_scaler.pkl
      - <Color>_feature_weights.pkl
      - <Color>_scaler_weighted.pkl
      - <Color>_pca_weighted.pkl
      - <Color>_svm_model.pkl

    Args:
        spectral_array (array-like): 1D array of raw spectral values.
        color_label (str): filament color (e.g. 'black', 'white').

    Returns:
        str: material prediction (e.g., 'PLA', 'PET', 'ASA').
    """
    # Ensure numpy array and correct shape
    X = np.array(spectral_array, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Base model directory
    model_dir = './models_per_color'

    # Load global material encoder
    encoder_path = os.path.join(model_dir, 'material_encoder.pkl')
    material_encoder = joblib.load(encoder_path)

    # Normalize color label and build file prefix
    clr = color_label.strip().capitalize()
    prefix = os.path.join(model_dir, clr)

    # Load pipeline components
    scaler           = joblib.load(f'{prefix}_scaler.pkl')
    feature_weights  = joblib.load(f'{prefix}_feature_weights.pkl')
    scaler_weighted  = joblib.load(f'{prefix}_scaler_weighted.pkl')
    pca              = joblib.load(f'{prefix}_pca_weighted.pkl')
    svm_model        = joblib.load(f'{prefix}_svm_model.pkl')

    # Pipeline: scale, weight, scale, PCA, SVM
    X_scaled     = scaler.transform(X)
    X_weighted   = X_scaled * feature_weights
    X_wt_scaled  = scaler_weighted.transform(X_weighted)
    X_pca        = pca.transform(X_wt_scaled)
    y_pred       = svm_model.predict(X_pca)

    # Return decoded label
    return material_encoder.inverse_transform(y_pred)[0]
