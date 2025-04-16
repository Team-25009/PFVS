#!/usr/bin/env python3
import os
import numpy as np
import joblib
import logging

# ——— Setup logger ———
logger = logging.getLogger("pfvs.predict")
logger.setLevel(logging.DEBUG)  # show DEBUG logs
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)

# … inside predict_material.py …

# Map single‑letter to full names used in filenames
COLOR_MAP = {
    'B': 'Blue',
    'G': 'Green',
    'R': 'Red',
    'K': 'Black',
    'W': 'White'
}

def load_color_pipeline(color_label, model_dir=None):
    """Load model components for either a one‑letter code or full color name."""
    cl = color_label.strip()              # remove leading/trailing spaces
    if len(cl) == 1:
        # single‑letter → map
        code = cl.upper()
        if code not in COLOR_MAP:
            raise ValueError(f"Unknown color code: {color_label!r}")
        full = COLOR_MAP[code]
    else:
        # assume full name, capitalize to match filenames
        full = cl.capitalize()
        if full not in COLOR_MAP.values():
            raise ValueError(f"Unknown color name: {color_label!r}")

    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), 'models_per_color')

    comp = {}
    to_load = [
        ('scaler',    f'{full}_scaler.pkl'),
        ('weights',   f'{full}_feature_weights.pkl'),
        ('scaler_w',  f'{full}_scaler_weighted.pkl'),
        ('pca',       f'{full}_pca_weighted.pkl'),
        ('svm',       f'{full}_svm_model.pkl'),
        ('encoder',   'material_encoder.pkl')
    ]
    for key, fname in to_load:
        path = os.path.join(model_dir, fname)
        logger.debug(f"Loading {key} from {path}")
        comp[key] = joblib.load(path)
        logger.debug(f"  → {key} loaded (type={type(comp[key]).__name__})")
    return comp


def predict_material(spectral_data, color_label):
    """
    Predict filament material for one 18‑dim spectral sample and a color code.
    """
    # Convert and validate
    X = np.array(spectral_data, dtype=float).reshape(1, -1)
    logger.debug(f"Raw input shape: {X.shape}")
    if X.shape[1] != 18:
        raise ValueError("Need exactly 18 spectral channel values.")
    
    # Load pipeline
    pipe = load_color_pipeline(color_label)
    
    # 1) scale raw
    Xs = pipe['scaler'].transform(X)
    logger.debug(f"After scaler.transform: {Xs.shape}, sample[0]={Xs[0,:3]}…")
    
    # 2) weight features
    Xw = Xs * pipe['weights']
    logger.debug(f"After weighting: {Xw.shape}, weighted[0]={Xw[0,:3]}…")
    
    # 3) re‑scale weighted
    Xsw = pipe['scaler_w'].transform(Xw)
    logger.debug(f"After scaler_weighted.transform: {Xsw.shape}, sample[0]={Xsw[0,:3]}…")
    
    # 4) PCA
    Xp = pipe['pca'].transform(Xsw)
    logger.debug(f"After PCA.transform: {Xp.shape}, components={pipe['pca'].n_components_}")
    
    # 5) Predict
    y_enc = pipe['svm'].predict(Xp)
    logger.debug(f"SVM.predict returned encoded label: {y_enc}")
    material = pipe['encoder'].inverse_transform(y_enc.astype(int))[0]
    logger.info(f"Color={color_label}, Predicted material={material}")
    return material

if __name__ == "__main__":
    import pandas as pd
    from io import StringIO

    # map single‑letter → full name
    COLOR_MAP = {'B': 'Blue', 'G': 'Green', 'R': 'Red', 'K': 'Black', 'W': 'White'}

    # hard‑coded CSV without 'Scan' column
    # ——— Hard‑coded test scans CSV (no 'Scan' column) ———
    csv_data = """Color,Material,Value1,Value2,Value3,Value4,Value5,Value6,Value7,Value8,Value9,Value10,Value11,Value12,Value13,Value14,Value15,Value16,Value17,Value18
    Red,PLA,472,103,295,87,166,207,82,179,161,152,36,52,19,14,29,76,21,17
    Red,PLA,470,102,294,87,165,206,82,179,161,152,36,52,18,14,29,76,20,17
    Red,PLA,469,102,293,87,165,206,82,179,160,152,36,52,18,14,29,76,21,17
    Red,ASA,468,111,307,99,175,212,95,206,157,326,37,68,19,14,29,76,25,20
    Red,ASA,467,111,307,99,176,212,95,207,158,328,36,68,19,14,29,77,25,20
    Red,ASA,465,111,307,99,175,212,95,207,157,326,38,68,19,14,29,76,25,20
    Red,PETG,465,105,300,88,166,206,91,184,158,246,37,65,19,14,29,76,22,19
    Red,PETG,467,105,301,89,167,207,91,184,158,246,37,65,19,14,29,76,22,19
    Red,PETG,468,106,301,89,167,207,91,185,158,246,37,65,19,14,29,76,22,19
    Green,PLA,599,150,422,218,502,427,177,194,152,97,35,32,17,13,29,78,25,18
    Green,PLA,599,149,422,218,501,426,177,194,152,97,35,32,18,13,29,78,25,18
    Green,PLA,599,149,421,217,502,426,177,194,152,97,35,32,18,13,29,78,25,18
    Green,ASA,541,125,417,201,376,425,180,168,151,72,34,32,18,13,28,76,22,16
    Green,ASA,541,125,417,201,376,424,180,168,151,72,35,32,18,13,28,77,22,16
    Green,ASA,542,125,417,201,375,424,179,169,151,72,35,32,18,13,28,77,22,16
    Green,PETG,642,127,381,232,426,409,180,175,149,75,34,30,17,13,28,75,20,17
    Green,PETG,644,128,382,232,427,409,180,175,149,75,34,30,17,13,28,75,20,17
    Green,PETG,646,128,383,233,427,409,180,175,149,75,34,30,17,13,28,75,20,17
    Blue,PLA,609,178,553,166,226,230,82,97,150,66,35,30,17,13,28,76,17,16
    Blue,PLA,607,178,553,166,226,230,82,97,150,66,35,30,17,13,28,76,17,16
    Blue,PLA,606,178,553,166,225,229,82,97,150,66,35,30,17,13,28,76,17,16
    Blue,ASA,713,187,499,154,216,230,84,101,149,65,35,33,18,13,28,75,15,16
    Blue,ASA,713,186,498,154,216,229,84,101,149,65,35,33,18,13,28,75,15,16
    Blue,ASA,713,186,498,154,216,229,84,101,149,65,35,33,17,13,28,75,15,16
    Blue,PETG,1056,357,1003,384,439,322,118,132,150,77,35,30,18,13,29,77,24,18
    Blue,PETG,1056,358,1004,385,440,323,119,132,151,77,35,30,17,13,29,77,23,18
    Blue,PETG,1058,358,1004,385,439,323,119,132,150,77,35,30,18,13,28,77,24,18
    Black,PLA,381,90,287,87,160,202,80,103,149,62,34,24,17,13,28,74,11,15
    Black,PLA,380,89,286,86,160,201,80,103,149,62,34,24,17,13,28,74,11,15
    Black,PLA,380,89,286,86,159,201,80,103,149,62,34,24,17,13,28,74,11,15
    Black,ASA,481,101,301,90,163,205,81,105,150,64,35,24,17,13,28,75,11,15
    Black,ASA,484,102,302,90,164,205,81,105,150,64,35,24,17,13,28,75,11,15
    Black,ASA,488,103,304,91,165,206,81,106,150,64,35,24,17,13,28,75,11,15
    Black,PETG,430,97,305,89,171,220,75,95,150,58,34,23,17,13,28,74,11,15
    Black,PETG,432,98,307,89,172,220,75,95,149,58,34,23,17,13,28,74,11,15
    Black,PETG,433,98,307,89,172,221,75,95,150,58,35,23,17,13,28,74,11,15
    White,PLA,1990,570,1334,604,852,727,426,565,165,286,38,73,19,14,29,78,33,27
    White,PLA,1996,571,1335,606,851,726,425,565,165,285,37,73,19,14,29,78,34,27
    White,PLA,1996,570,1334,605,852,727,426,566,165,286,38,73,19,14,29,78,33,27
    White,ASA,1810,600,1531,626,966,944,505,613,169,282,38,96,19,14,30,80,42,28
    White,ASA,1813,601,1532,627,965,944,505,612,169,282,39,96,20,14,30,80,42,28
    White,ASA,1812,601,1531,627,966,944,505,613,169,282,39,96,20,14,30,80,42,28
    White,PETG,1933,625,1527,651,964,887,491,626,168,304,38,86,19,14,30,79,38,28
    White,PETG,1937,625,1527,651,964,886,490,625,168,304,38,86,19,14,30,79,38,28
    White,PETG,1938,625,1526,651,963,886,490,625,168,304,38,86,19,14,30,79,38,28
    """
    df = pd.read_csv(StringIO(csv_data.strip()))
    logger.setLevel(logging.INFO)  # only INFO+ messages

    predictions = []
    for i, row in df.iterrows():
        color_raw = row['Color'].strip()
        # determine which model file we’re loading
        if len(color_raw) == 1:
            model_color = COLOR_MAP[color_raw.upper()]
        else:
            model_color = color_raw.capitalize()

        scan = [row[f'Value{j}'] for j in range(1, 19)]
        true_mat = row['Material']
        # predict
        pred_mat = predict_material(scan, color_raw)
        # record
        predictions.append(pred_mat)

        print(f"Row {i+1}: Using {model_color} model → True: {true_mat}, Predicted: {pred_mat}")

    # final summary
    print("\nAll predictions:")
    for i, pred in enumerate(predictions, start=1):
        print(f"  Row {i}: {pred}")

    import matplotlib.pyplot as plt

# assume you still have:
# df (cols: Color, Material, Value1…Value18)
# predictions (list of predicted materials, same order)

# attach predictions to df
df = df.copy()
df['Pred'] = predictions

for color_raw, group in df.groupby('Color'):
    code = color_raw.strip()
    # load just this color's pipeline
    pipe = load_color_pipeline(code)

    # 1) pull out spectral matrix
    X = group[[f'Value{i}' for i in range(1,19)]].values.astype(float)

    # 2) transform all the way to PCA
    Xs  = pipe['scaler'].transform(X)
    Xw  = Xs * pipe['weights']
    Xsw = pipe['scaler_w'].transform(Xw)
    Xp  = pipe['pca'].transform(Xsw)   # shape = (n_samples, 2)

    # 3) build a meshgrid over the range of Xp
    margin = 1
    x_min, x_max = Xp[:,0].min() - margin, Xp[:,0].max() + margin
    y_min, y_max = Xp[:,1].min() - margin, Xp[:,1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = pipe['svm'].predict(grid).reshape(xx.shape)

    # 4) plot
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # color‐encode the sample points by their predicted material index
    # first map predicted material str → integer via encoder
    y_idx = pipe['encoder'].transform(group['Pred'].values)
    sc = plt.scatter(
        Xp[:,0], Xp[:,1],
        c=y_idx,
        cmap=plt.cm.coolwarm,
        edgecolor='k',
        s=80
    )
    plt.colorbar(sc, label="Encoded material")
    plt.title(f"{code.capitalize()} scans on {code.capitalize()} boundary")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.show()
