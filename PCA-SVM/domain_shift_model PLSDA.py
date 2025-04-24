import os
import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Suppress sklearn cross-validation warnings
warnings.filterwarnings("ignore", category=UserWarning)


def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def coral_align(source, target):
    cov_source = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-6
    cov_target = np.cov(target, rowvar=False) + np.eye(source.shape[1]) * 1e-6
    U_src, S_src, _ = np.linalg.svd(cov_source)
    whiten = U_src @ np.diag(1.0 / np.sqrt(S_src)) @ U_src.T
    src_white = (source - np.mean(source, axis=0)) @ whiten
    U_tgt, S_tgt, _ = np.linalg.svd(cov_target)
    recolor = U_tgt @ np.diag(np.sqrt(S_tgt)) @ U_tgt.T
    aligned = src_white @ recolor + np.mean(target, axis=0)
    return aligned


def compute_feature_weights(importances, min_scale=0.01, max_scale=1.0):
    ptp = importances.ptp()
    if ptp == 0:
        return np.ones_like(importances)
    return min_scale + (importances - importances.min()) / ptp * (max_scale - min_scale)


def extract_info_from_filename(filename):
    base = os.path.basename(filename)
    return base[:3], base[3]


def load_environment_data(folder_path, selected_files=None, has_header=False):
    folder = Path(folder_path)
    all_spectra, all_materials, all_colors = [], [], []
    if selected_files:
        file_paths = [folder / f for f in selected_files]
    else:
        file_paths = list(folder.glob("*.csv"))

    for fp in file_paths:
        if not fp.exists():
            print(f"Warning: {fp} not found, skipping.")
            continue
        try:
            if has_header:
                df = pd.read_csv(fp)
                df.columns = df.columns.str.lower()
                if 'color' in df and 'material' in df:
                    colors = df['color'].astype(str).str.strip().str.capitalize()
                    materials = df['material'].astype(str).str.strip()
                else:
                    mat, col = extract_info_from_filename(fp.name)
                    colors = pd.Series([col]*len(df))
                    materials = pd.Series([mat]*len(df))
                spectral = df.filter(regex='(?i)^value').apply(pd.to_numeric, errors='coerce')
                valid = spectral.dropna().index
                spectral = spectral.loc[valid]
                colors = colors.loc[valid]
                materials = materials.loc[valid]
            else:
                df = pd.read_csv(fp, header=None)
                if df.shape[1] < 3:
                    print(f"Skipping {fp}: insufficient cols.")
                    continue
                num1 = is_number(df.iloc[0,0])
                num2 = is_number(df.iloc[0,1])
                if num1 and num2:
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
            data = spectral.values
            valid_meta = [(c and m and c.lower()!='nan') for c,m in zip(colors, materials)]
            data = data[valid_meta]
            all_spectra.append(data)
            all_materials.extend([materials.iloc[i] for i in np.where(valid_meta)[0]])
            all_colors.extend([colors.iloc[i] for i in np.where(valid_meta)[0]])
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    if not all_spectra:
        return np.empty((0,0)), np.array([]), np.array([])
    return np.vstack(all_spectra), np.array(all_materials), np.array(all_colors)


def main():
    base = Path(__file__).parent
    cfg = {
        'initial': {'path': base/'data_old', 'weight':1.0, 'header':False},
        'current': {'path': base/'data_current', 'weight':1.0, 'header':False}
    }
    env = {}
    for name,info in cfg.items():
        X, mats, cols = load_environment_data(info['path'], has_header=info['header'])
        env[name] = {'X':X, 'materials':mats, 'colors':cols, 'w':info['weight']}
        print(f"Loaded {X.shape[0]} samples for {name}")

    all_mats = np.concatenate([env[e]['materials'] for e in env])
    le = LabelEncoder(); le.fit(all_mats)
    os.makedirs('models_per_color', exist_ok=True)
    joblib.dump(le, 'models_per_color/material_encoder.pkl')

    all_cols = np.concatenate([env[e]['colors'] for e in env])
    for col in np.unique(all_cols):
        if not col or col.lower()=='nan':
            continue
        print(f"\n=== Color: {col} ===")
        X_list, y_list, env_lab = [], [], []
        for name in env:
            mask = env[name]['colors']==col
            Xi = env[name]['X'][mask]
            yi = env[name]['materials'][mask]
            if Xi.shape[0]>0:
                X_list.append((name,Xi)); y_list.append(yi)
                env_lab.extend([name]*Xi.shape[0])
        if not X_list: continue
        X_raw = np.vstack([x for _,x in X_list]).astype(float)
        y_labels = np.concatenate(y_list)
        y_enc = le.transform(y_labels)
        samp_w = np.concatenate([np.full(x.shape[0], cfg[n]['weight']) for n,x in X_list])

        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_raw)
        by_env = {}
        idx=0
        for n,x in X_list:
            cnt = x.shape[0]
            by_env[n]=X_scaled[idx:idx+cnt]; idx+=cnt
        if 'initial' in by_env and 'current' in by_env and by_env['current'].size>0:
            by_env['initial'] = coral_align(by_env['initial'], by_env['current'])
        X_aligned = np.vstack([by_env[n] for n,_ in X_list])

        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_aligned, y_enc, sample_weight=samp_w)
        imp = rf.feature_importances_
        weights = compute_feature_weights(imp)

        X_weighted = X_aligned * weights
        scaler_w = StandardScaler(); X_w_sc = scaler_w.fit_transform(X_weighted)
        pca_w = PCA(n_components=2); X_w_pca = pca_w.fit_transform(X_w_sc)

        # PLS-DA
        Y_oh = pd.get_dummies(y_enc).values
        pls = PLSRegression(n_components=2)
        pls.fit(X_w_sc, Y_oh)
        X_scores = pls.transform(X_w_sc)  # use full 2D array of scores
        clf_pls = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        clf_pls.fit(X_scores, y_enc, sample_weight=samp_w)
        print("PLS-DA Report:")
        y_pred_pls = clf_pls.predict(X_scores)
        print(classification_report(y_enc, y_pred_pls, target_names=le.classes_))

        # SVM on PCA scores
        idx_all = np.arange(len(y_enc))
        tr, te = train_test_split(idx_all, test_size=0.3, random_state=42, stratify=y_enc)
        X_tr, X_te = X_w_pca[tr], X_w_pca[te]
        y_tr, y_te = y_enc[tr], y_enc[te]
        w_tr, w_te = samp_w[tr], samp_w[te]

        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
        svm.fit(X_tr, y_tr, sample_weight=w_tr)
        y_tr_pred = svm.predict(X_tr)
        y_te_pred = svm.predict(X_te)
        print("SVM Train:")
        print(classification_report(y_tr, y_tr_pred, target_names=le.classes_))
        print("SVM Test:")
        print(classification_report(y_te, y_te_pred, target_names=le.classes_))

        comps = {'scaler':scaler, 'rf':rf, 'weights':weights,
                 'scaler_w':scaler_w, 'pca_w':pca_w, 'pls':pls,
                 'clf_pls':clf_pls, 'svm':svm, 'encoder':le}
        joblib.dump(comps, f'models_per_color/{col}_components.pkl')
        print(f"Saved models for {col}")

if __name__ == '__main__':
    main()
