import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Helper: Check if a value can be converted to float.
# -------------------------
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# -------------------------
# Utility Function to Extract Metadata from Filename
# -------------------------
def extract_info_from_filename(filename):
    base = os.path.basename(filename)
    material = base[:3]
    color = base[3]
    return material, color

# -------------------------
# Data Loading Function
# -------------------------
def load_environment_data(env_name, folder_path, selected_files=None, has_header=False):
    folder = Path(folder_path)
    all_spectra, all_materials, all_colors = [], [], []

    files = selected_files if selected_files else list(folder.glob("*.csv"))
    for f in files:
        fp = folder / f if not isinstance(f, Path) else f
        if not fp.exists():
            print(f"Warning: {fp} not found. Skipping.")
            continue
        try:
            if has_header:
                df = pd.read_csv(fp, header=0)
                df.columns = df.columns.str.lower()
                if 'color' in df.columns and 'material' in df.columns:
                    colors = df['color'].astype(str).str.strip().str.capitalize()
                    materials = df['material'].astype(str).str.strip()
                else:
                    mat, col = extract_info_from_filename(fp.name)
                    colors = pd.Series([col]*len(df))
                    materials = pd.Series([mat]*len(df))
                spectral = df.filter(regex=r'(?i)^value').apply(pd.to_numeric, errors='coerce')
                valid = spectral.dropna().index
                spectral = spectral.loc[valid]
                colors = colors.loc[valid]
                materials = materials.loc[valid]
            else:
                df = pd.read_csv(fp, header=None)
                if df.shape[1] < 3:
                    continue
                c0_num = is_number(df.iloc[0,0])
                c1_num = is_number(df.iloc[0,1])
                if c0_num and c1_num:
                    mat, col = extract_info_from_filename(fp.name)
                    colors = pd.Series([col]*len(df))
                    materials = pd.Series([mat]*len(df))
                    spectral = df.apply(pd.to_numeric, errors='coerce').dropna()
                else:
                    colors = df.iloc[:,0].astype(str).str.strip().str.capitalize()
                    materials = df.iloc[:,1].astype(str).str.strip()
                    spectral = df.iloc[:,2:].apply(pd.to_numeric, errors='coerce')
                    valid = spectral.dropna().index
                    spectral = spectral.loc[valid]
                    colors = colors.loc[valid]
                    materials = materials.loc[valid]
            if spectral.empty:
                continue
            arr = spectral.values
            all_spectra.append(arr)
            n = arr.shape[0]
            all_colors.extend(colors[:n].tolist())
            all_materials.extend(materials[:n].tolist())
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    if not all_spectra:
        return np.empty((0,0)), np.array([]), np.array([])
    return np.vstack(all_spectra), np.array(all_materials), np.array(all_colors, dtype=str)

# -------------------------
# Weighting Helper Function
# -------------------------
def compute_feature_weights(importances, min_scale=0.01, max_scale=1.0):
    ptp = importances.ptp()
    if ptp == 0:
        return np.ones_like(importances)
    return min_scale + (importances - importances.min()) / ptp * (max_scale - min_scale)

# -------------------------
# Main PLS-DA Pipeline WITHOUT CORAL
# -------------------------
def main():
    base = os.path.dirname(__file__)
    cfg = {
        'initial': dict(path=os.path.join(base,'data_old'), weight=1.0),
        'current': dict(path=os.path.join(base,'data_current'), weight=10.0)
    }
    env_data = {}
    for name, c in cfg.items():
        X, mats, cols = load_environment_data(name, c['path'], has_header=False)
        env_data[name] = dict(X=X, mats=mats, cols=cols, weight=c['weight'])
        print(f"Loaded {X.shape[0]} samples for {name}")

    all_mats = np.concatenate([d['mats'] for d in env_data.values()])
    mat_enc = LabelEncoder().fit(all_mats)
    os.makedirs('models_plsda', exist_ok=True)
    joblib.dump(mat_enc, 'models_plsda/material_encoder.pkl')

    unique_colors = np.unique(np.concatenate([d['cols'] for d in env_data.values()]))
    for color in unique_colors:
        if not color or color.lower()=='nan':
            continue
        print(f"\n=== Color: {color} ===")

        X_list, y_list, w_list = [], [], []
        for name, d in env_data.items():
            mask = (d['cols'] == color)
            Xi, yi = d['X'][mask], d['mats'][mask]
            if Xi.shape[0] > 1:
                X_list.append(Xi)
                y_list.append(yi)
                w_list.append(np.full(Xi.shape[0], d['weight']))
        if not X_list:
            print(" no data → skip")
            continue

        # Combine and encode
        X_raw = np.vstack(X_list).astype(float)
        y_comb = np.concatenate(y_list)
        y_enc = mat_enc.transform(y_comb)
        sample_weights = np.concatenate(w_list)

        # Scale data
        scaler = StandardScaler().fit(X_raw)
        Xs = scaler.transform(X_raw)

        # NO CORAL alignment: use Xs directly
        X_aligned = Xs

        # Feature importances and weights
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_aligned, y_enc, sample_weight=sample_weights)
        fw = compute_feature_weights(rf.feature_importances_)

        # Apply feature weights
        Xw = X_aligned * fw

        # PLS-DA projection
        pls = PLSRegression(n_components=2)
        Xpls = pls.fit_transform(Xw, pd.get_dummies(y_enc).values)[0]

        # Train/Test split
        idxs = np.arange(len(y_enc))
        tr, te = train_test_split(idxs, test_size=0.3, random_state=42, stratify=y_enc)
        Xtr, Xte = Xpls[tr], Xpls[te]
        ytr, yte = y_enc[tr], y_enc[te]

        # Logistic Regression
        clf = LogisticRegression(class_weight='balanced', max_iter=500)
        clf.fit(Xtr, ytr)

        # Reports
        print("PLS-DA Train:")
        print(classification_report(ytr, clf.predict(Xtr), target_names=mat_enc.classes_))
        print("PLS-DA Test:")
        print(classification_report(yte, clf.predict(Xte), target_names=mat_enc.classes_))

        # Save pipeline
        joblib.dump({
            'scaler': scaler,
            'rf': rf,
            'feature_weights': fw,
            'pls': pls,
            'clf': clf,
            'mat_enc': mat_enc
        }, f'models_plsda/{color}_plsda.pkl')
        print(f"Saved PLS-DA pipeline for {color}")

if __name__ == '__main__':
    main()
