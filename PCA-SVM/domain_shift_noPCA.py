import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC

# -------------------------
# Feature Alignment Function (CORAL)
# -------------------------
def coral_align(source, target):
    """
    Aligns the source domain features to the target domain using a CORAL-like approach.
    Computes a whitening transformation on the source and recolors it with the target's covariance structure.
    """
    cov_source = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-6
    cov_target = np.cov(target, rowvar=False) + np.eye(source.shape[1]) * 1e-6

    U_src, S_src, _ = np.linalg.svd(cov_source)
    whiten_matrix = U_src @ np.diag(1.0 / np.sqrt(S_src)) @ U_src.T
    source_whitened = (source - np.mean(source, axis=0)) @ whiten_matrix

    U_tgt, S_tgt, _ = np.linalg.svd(cov_target)
    color_matrix = U_tgt @ np.diag(np.sqrt(S_tgt)) @ U_tgt.T
    source_aligned = source_whitened @ color_matrix

    source_aligned += np.mean(target, axis=0)
    return source_aligned

# -------------------------
# Weighting Helper Function
# -------------------------
def compute_feature_weights(importances, min_scale=0.8, max_scale=1.2):
    """
    Linearly scale the importances to a desired range [min_scale, max_scale].
    If the range of importances is zero, returns an array of ones.
    """
    ptp = importances.ptp()  # peak-to-peak (max - min)
    if ptp == 0:
        return np.ones_like(importances)
    weights = min_scale + (importances - importances.min()) / ptp * (max_scale - min_scale)
    return weights

# -------------------------
# Utility Functions
# -------------------------
def extract_info_from_filename(filename):
    """
    For old files (without header): extract the material and color from the filename.
    Assumes the first three characters are the material and the fourth is the color.
    Example: "ASABfulldatabase.csv" → material = "ASA", color = "B".
    """
    base = os.path.basename(filename)
    material = base[:3]
    color = base[3]
    return material, color

def load_environment_data(env_name, data_dir, selected_files, has_header=False):
    """
    Loads spectral data for one environment.
    
    Parameters:
      - env_name: Environment name.
      - data_dir: Directory that contains the CSV files.
      - selected_files: List of CSV filenames.
      - has_header: True if the files include a header row.
    
    Returns a tuple (spectra, materials, colors) where:
      - spectra: NumPy array of floats containing only spectral values.
      - materials: Array of material labels.
      - colors: Array of color labels (strings).
    """
    all_spectra = []
    all_materials = []
    all_colors = []
    
    for filename in selected_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            if has_header:
                # New format: with header "Color,Material,Scan,Value1,..."
                df = pd.read_csv(file_path, header=0)
                color = str(df.loc[0, 'Color'])
                material = str(df.loc[0, 'Material'])
                spectral_data = df.filter(regex='^Value').astype(float).values
            else:
                # Old format: no header; extract metadata from filename.
                df = pd.read_csv(file_path, header=None)
                material, color = extract_info_from_filename(filename)
                spectral_data = df.values.astype(float)
            all_spectra.append(spectral_data)
            all_materials.extend([material] * spectral_data.shape[0])
            all_colors.extend([color] * spectral_data.shape[0])
        else:
            print(f"Warning: File {file_path} not found. Skipping.")
    
    if all_spectra:
        spectra_array = np.vstack(all_spectra)
    else:
        spectra_array = np.empty((0, 0))
    
    return spectra_array, np.array(all_materials), np.array(all_colors, dtype=str)

# -------------------------
# Main Training and Analysis Pipeline (No PCA)
# -------------------------
def main():
    base_dir = os.path.dirname(__file__)
    
    # CONFIGURATION: Use "data_old" for initial data (new-format CSVs) and "data_current" for new scans.
    datasets = {
        "initial": {
            "data_dir": os.path.join(base_dir, 'data_old'),
            "files": [
                "PLAGfulldatabase.csv", "ASAGfulldatabase.csv", "PETGfulldatabase.csv",
                "PLAKfulldatabase.csv", "ASAKfulldatabase.csv", "PETKfulldatabase.csv",
                "PLABfulldatabase.csv", "ASABfulldatabase.csv", "PETBfulldatabase.csv",
                "PLAWfulldatabase.csv", "ASAWfulldatabase.csv", "PETWfulldatabase.csv",
                "PLARfulldatabase.csv", "ASARfulldatabase.csv", "PETRfulldatabase.csv"
            ],
            "weight": 1.0,
            "has_header": True
        },
        "current": {
            "data_dir": os.path.join(base_dir, 'data_current'),
            "files": [
                # If you have any current files, add them here.
            ],
            "weight": 2.0,
            "has_header": True
        }
    }
    
    # Load data for each environment.
    env_data = {}
    for env in datasets:
        cfg = datasets[env]
        spectra, materials, colors = load_environment_data(
            env, cfg["data_dir"], cfg["files"], has_header=cfg.get("has_header", False)
        )
        env_data[env] = {"spectra": spectra, "materials": materials, "colors": colors, "weight": cfg["weight"]}
        print(f"Loaded {len(spectra)} samples for environment: {env}")
    
    # Create a common label encoder using all material labels.
    all_materials = np.concatenate([env_data[e]["materials"] for e in env_data])
    material_encoder = LabelEncoder()
    material_encoder.fit(all_materials)
    os.makedirs('./models_per_color', exist_ok=True)
    joblib.dump(material_encoder, './models_per_color/material_encoder.pkl')
    
    # Identify unique colors across environments.
    all_colors_combined = np.concatenate([env_data[e]["colors"] for e in env_data])
    unique_colors = np.unique(all_colors_combined)
    
    models_per_color = {}
    
    for col in unique_colors:
        print(f"\n=== Training and Analysis for Color: {col} ===")
        X_env_list = []
        y_env_list = []
        for env in env_data:
            idx = np.where(env_data[env]["colors"] == col)[0]
            X_env = env_data[env]["spectra"][idx]
            y_env = env_data[env]["materials"][idx]
            if X_env.shape[0] > 0:
                X_env_list.append((env, X_env))
                y_env_list.append((env, y_env))
            else:
                print(f"No samples found for color {col} in environment {env}")
        if not X_env_list:
            continue
        
        # Combine raw spectral data.
        X_combined_raw = np.vstack([x for (_, x) in X_env_list]).astype(float)
        scaler = StandardScaler()
        X_combined_scaled = scaler.fit_transform(X_combined_raw)
        
        # Split scaled data back by environment.
        X_scaled_by_env = {}
        start = 0
        for env, X_env in X_env_list:
            n_samples = X_env.shape[0]
            X_scaled_by_env[env] = X_combined_scaled[start:start+n_samples]
            start += n_samples
        
        # Combine and encode labels.
        y_combined_labels = np.concatenate([y for (_, y) in y_env_list])
        y_combined_encoded = material_encoder.transform(y_combined_labels)
        y_combined_final = y_combined_encoded
        
        # Optionally perform CORAL alignment if current data exists.
        if "initial" in X_scaled_by_env and "current" in X_scaled_by_env and X_scaled_by_env["current"].shape[0] > 0:
            X_initial_aligned = coral_align(X_scaled_by_env["initial"], X_scaled_by_env["current"])
            X_scaled_by_env["initial"] = X_initial_aligned
        
        X_combined_aligned = []
        sample_weights = []
        for env in X_scaled_by_env:
            X_data = X_scaled_by_env[env]
            X_combined_aligned.append(X_data)
            sample_weights.append(np.full(X_data.shape[0], datasets[env]["weight"]))
        X_combined_aligned = np.vstack(X_combined_aligned)
        sample_weights = np.concatenate(sample_weights)
        
        # ======================================
        # 1. Random Forest Feature Importance on Original Data
        # ======================================
        rf_original = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf_original.fit(X_combined_aligned, y_combined_final, sample_weight=sample_weights)
        importances = rf_original.feature_importances_
        num_features = X_combined_aligned.shape[1]
        feature_names = [f"Value{i+1}" for i in range(num_features)]
        sorted_idx = np.argsort(importances)[::-1]
        print("Feature importances (original feature space):")
        for idx in sorted_idx:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
        
        # Compute feature weight vector.
        weights = compute_feature_weights(importances)
        print("Computed feature weight vector for color", col, ":")
        for i, w in enumerate(weights):
            print(f"  {feature_names[i]}: {w:.4f}")
        
        # ======================================
        # 2. SVM Classification Using Weighted Features (No PCA)
        # ======================================
        # Apply the computed weights to the aligned data (column-wise multiplication).
        X_weighted = X_combined_aligned * weights
        
        # Re-scale the weighted features.
        scaler_weighted = StandardScaler()
        X_weighted_scaled = scaler_weighted.fit_transform(X_weighted)
        
        # Split data for SVM training.
        X_train_w, X_test_w, y_train_w, y_test_w, w_train_w, w_test_w = train_test_split(
            X_weighted_scaled, y_combined_final, sample_weights,
            test_size=0.3, random_state=42, stratify=y_combined_final
        )
        
        # Train SVM on full (weighted) features.
        svm_model = SVC(kernel='rbf', C=0.5, gamma='scale', class_weight='balanced', random_state=42)
        print("Training the SVM model on weighted features (no PCA)...")
        svm_model.fit(X_train_w, y_train_w, sample_weight=w_train_w)
        
        y_train_pred_svm = svm_model.predict(X_train_w)
        y_test_pred_svm = svm_model.predict(X_test_w)
        print("SVM Training Performance (weighted features, no PCA):")
        try:
            print(classification_report(y_train_w, y_train_pred_svm, target_names=material_encoder.classes_))
        except ValueError as e:
            print("Skipping SVM training classification report due to:", e)
        print("SVM Test Performance (weighted features, no PCA):")
        try:
            print(classification_report(y_test_w, y_test_pred_svm, target_names=material_encoder.classes_))
        except ValueError as e:
            print("Skipping SVM test classification report due to:", e)
        
        ConfusionMatrixDisplay.from_predictions(
            y_test_w, y_test_pred_svm,
            display_labels=material_encoder.classes_, cmap='Oranges'
        )
        plt.title(f"SVM: Test Confusion Matrix for Color {col}")
        plt.show()
        
        # ======================================
        # 3. (Optional) Learning Curve Plotting for SVM with Weighted Features
        # ======================================
        svm_for_curve = SVC(kernel='rbf', C=0.5, gamma='scale', class_weight='balanced', random_state=42)
        train_sizes, train_scores, val_scores = learning_curve(
            svm_for_curve, X_weighted_scaled, y_combined_final, cv=5, scoring='accuracy',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, random_state=42
        )
        plt.figure(figsize=(8,6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Accuracy')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Accuracy')
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.title(f"SVM Learning Curve (Weighted) for Color {col} (No PCA)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Save model components for this color.
        model_components = {
            'scaler': scaler,
            'scaler_weighted': scaler_weighted,
            'svm_model': svm_model,
            'material_encoder': material_encoder,
            'rf_original': rf_original,
            'feature_weights': weights
        }
        models_per_color[col] = model_components
        joblib.dump(scaler, f'./models_per_color/{col}_scaler.pkl')
        joblib.dump(scaler_weighted, f'./models_per_color/{col}_scaler_weighted.pkl')
        joblib.dump(svm_model, f'./models_per_color/{col}_svm_model.pkl')
        print(f"Model components for color {col} saved.")
    
if __name__ == "__main__":
    main()
