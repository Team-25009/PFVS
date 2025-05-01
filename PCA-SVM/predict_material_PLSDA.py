#!/usr/bin/env python3
"""
predict_material_plsda: Predict filament material using a PLS-DA model on a Raspberry Pi.

Usage:
    python predict_material_plsda.py <Color> <val1> ... <val18>
Example:
    python predict_material_plsda.py B 1159 537 1490 616 684 479 153 143 145 86 41 37 21 14 29 82 32 23
"""
import os
import argparse
import numpy as np
import joblib

__all__ = ["load_plsda_pipeline", "predict_material_plsda"]

# Map one-letter codes to full color names
COLOR_MAP = {
    'B': 'Blue',
    'G': 'Green',
    'R': 'Red',
    'K': 'Black',
    'W': 'White',
}

def load_plsda_pipeline(color_label):
    """
    Load the PLS-DA pipeline for the given filament color.
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

    model_path = os.path.join(
        os.path.dirname(__file__),
        'models_plsda',
        f"{color}_plsda.pkl"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PLS-DA model not found at {model_path}")

    return joblib.load(model_path)


def predict_material_plsda(spectral_data, color_label):
    """
    Predict material from spectral data using PLS-DA pipeline.
    """
    X = np.asarray(spectral_data, dtype=float).reshape(1, -1)
    pipe = load_plsda_pipeline(color_label)

    Xs  = pipe['scaler'].transform(X)
    Xw  = Xs * pipe['feature_weights']
    Xp  = pipe['pls'].transform(Xw)
    y   = pipe['clf'].predict(Xp)
    return pipe['mat_enc'].inverse_transform(y)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict filament material via PLS-DA model")
    parser.add_argument('color', help="Color code or name (e.g. B or Blue)")
    parser.add_argument('values', metavar='val', type=float, nargs=18,
                        help="18 spectral values")
    args = parser.parse_args()

    material = predict_material_plsda(args.values, args.color)
    print(material)

