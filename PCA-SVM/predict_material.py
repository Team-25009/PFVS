#!/usr/bin/env python3
"""
predict_material: Pipeline to predict filament material from spectral data.

Functions:
  - load_color_pipeline: load preprocessing and model artifacts per color
  - predict_material: scale, weight, reduce, and predict material label

Usage in code:
  from octoprint_pfvs.predict_material import predict_material
  material = predict_material(spectral_data, color_label)

Optional CLI:
  python predict_material.py --model-dir ./models_per_color --verbose Red "scan.csv"
"""
import os
import sys
import argparse
import numpy as np
import joblib

__all__ = ["load_color_pipeline", "predict_material"]

# mapping one-letter codes to full color names (match model filenames)
COLOR_MAP = {
    'B': 'Blue',
    'G': 'Green',
    'R': 'Red',
    'K': 'Black',
    'W': 'White',
}

def load_color_pipeline(color_label, model_dir=None):
    """
    Load preprocessing and model artifacts for the given color.

    Args:
        color_label (str): single-letter code or full color name.
        model_dir (str, optional): directory where the model files live.
            Defaults to './models_per_color' relative to this script.

    Returns:
        dict: keys ['scaler', 'weights', 'scaler_w', 'pca', 'svm', 'encoder']
    """
    label = color_label.strip()
    if len(label) == 1:
        code = label.upper()
        if code not in COLOR_MAP:
            raise ValueError(f"Unknown color code: {label!r}")
        color = COLOR_MAP[code]
    else:
        color = label.capitalize()
        if color not in COLOR_MAP.values():
            raise ValueError(f"Unknown color name: {label!r}")

    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), 'models_per_color')

    artifacts = {
        'scaler':    f'{color}_scaler.pkl',
        'weights':   f'{color}_feature_weights.pkl',
        'scaler_w':  f'{color}_scaler_weighted.pkl',
        'pca':       f'{color}_pca_weighted.pkl',
        'svm':       f'{color}_svm_model.pkl',
        'encoder':   'material_encoder.pkl'
    }
    pipeline = {}
    for key, fname in artifacts.items():
        path = os.path.join(model_dir, fname)
        pipeline[key] = joblib.load(path)

    return pipeline

def predict_material(spectral_data, color_label, model_dir=None, return_confidence=False):
    """
    Predict filament material from a single 18-dimensional spectral sample.

    Args:
        spectral_data (array-like): length-18 list or array of raw channel values.
        color_label (str): single-letter or full color name for pipeline selection.
        model_dir (str, optional): path to models directory.
        return_confidence (bool): if True, also return max prediction probability.

    Returns:
        str or (str, float): predicted material (and confidence if requested).
    """
    X = np.asarray(spectral_data, dtype=float)
    if X.ndim != 1 or X.size != 18:
        raise ValueError(f"spectral_data must be one-dimensional of length 18, got shape={X.shape}")
    X = X.reshape(1, -1)

    pipe = load_color_pipeline(color_label, model_dir)
    # 1) scale
    Xs = pipe['scaler'].transform(X)
    # 2) apply feature weights
    Xw = Xs * pipe['weights']
    # 3) re-scale weighted
    Xsw = pipe['scaler_w'].transform(Xw)
    # 4) PCA
    Xp = pipe['pca'].transform(Xsw)
    # 5) SVM predict
    y_enc = pipe['svm'].predict(Xp)
    material = pipe['encoder'].inverse_transform(y_enc.astype(int))[0]

    if return_confidence and hasattr(pipe['svm'], 'predict_proba'):
        probs = pipe['svm'].predict_proba(Xp)[0]
        conf = float(np.max(probs))
        return material, conf
    return material

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict filament material from spectral data CSV."
    )
    parser.add_argument('color', help="Color code or name (e.g. R or Red)")
    parser.add_argument('csv', help="Path to CSV with header Value1..Value18 columns.")
    parser.add_argument('--model-dir', help="Directory of model files", default=None)
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv)
    required = [f'Value{i}' for i in range(1, 19)]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Input CSV must contain columns: {required}")

    for idx, row in df.iterrows():
        scan = [row[f'Value{i}'] for i in range(1, 19)]
        mat = predict_material(scan, args.color, args.model_dir)
        print(f"Row {idx+1}: Predicted material = {mat}")
