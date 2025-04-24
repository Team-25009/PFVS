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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC

# Suppress sklearn cross-validation warnings (FitFailedWarning)
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
# Feature Alignment Function (CORAL)
# -------------------------
def coral_align(source, target):
    """
    Aligns the source domain features to the target domain using a CORAL-like approach.
    This performs a whitening transformation on the source data, then recolors it with the 
    covariance structure of the target.
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
def compute_feature_weights(importances, min_scale=0.01, max_scale=1.0):
    """
    Scales feature importances linearly into a desired range [min_scale, max_scale].
    If all importances are equal (i.e. peak-to-peak is 0), returns an array of ones.
    """
    ptp = importances.ptp()  # Difference between max and min
    if ptp == 0:
        return np.ones_like(importances)
    weights = min_scale + (importances - importances.min()) / ptp * (max_scale - min_scale)
    return weights

# -------------------------
# Utility Function to Extract Metadata from Filename
# -------------------------
def extract_info_from_filename(filename):
    """
    Extracts the material and color from a filename.
    Assumes the first three characters are the material and the fourth is the color.
    For example, "ASABfulldatabase.csv" leads to material "ASA" and color "B".
    """
    base = os.path.basename(filename)
    material = base[:3]
    color = base[3]
    return material, color

# -------------------------
# Data Loading Function
# -------------------------
def load_environment_data(env_name, folder_path, selected_files=None, has_header=False):
    """
    Loads spectral data from CSV files in a folder.
    
    For files with headers, expected columns include "color", "material", and spectral columns 
    starting with "value" (case-insensitive). For files without headers, the function attempts 
    to determine if each row holds its own metadata:
      - If the first two columns are numeric, it assumes no per-row metadata and uses the filename.
      - Otherwise, it assumes the first two columns hold per-row metadata (color in col 0 and material in col 1).
    
    To clean the data:
      - All spectral columns are converted to numeric (invalid entries become NaN).
      - Rows with NaN in the spectral data are dropped.
      - Rows with missing metadata for color or material are dropped.
      
    Returns:
      Tuple (spectra, materials, colors) where:
        spectra    : NumPy array of floats containing the spectral data.
        materials  : NumPy array of material labels.
        colors     : NumPy array of color labels (as strings).
    """
    folder = Path(folder_path)
    all_spectra = []
    all_materials = []
    all_colors = []
    
    if selected_files and len(selected_files) > 0:
        file_paths = [folder / filename for filename in selected_files]
    else:
        file_paths = list(folder.glob("*.csv"))
    
    for file_path in file_paths:
        if not file_path.exists():
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
        
        try:
            # Process files that have a header.
            if has_header:
                df = pd.read_csv(file_path, header=0)
                df.columns = df.columns.str.lower()
                # Determine color and material from columns if available.
                if 'color' in df.columns and 'material' in df.columns:
                    colors_data = df['color'].astype(str).str.strip().str.capitalize()
                    materials_data = df['material'].astype(str).str.strip()
                else:
                    material, color = extract_info_from_filename(file_path.name)
                    colors_data = pd.Series([str(color).strip().capitalize()] * len(df))
                    materials_data = pd.Series([material] * len(df))
                    
                # Convert spectral columns (those that start with "value") to numeric.
                spectral_df = df.filter(regex=r'(?i)^value')
                spectral_df = spectral_df.apply(pd.to_numeric, errors='coerce')
                # Drop rows that have NaN in any spectral column.
                valid_idx = spectral_df.dropna().index
                spectral_df = spectral_df.loc[valid_idx]
                colors_data = colors_data.loc[valid_idx]
                materials_data = materials_data.loc[valid_idx]
                if spectral_df.empty:
                    print(f"Skipping {file_path} (after cleaning): no valid spectral data.")
                    continue
                
            else:
                # Process files that have no header.
                df = pd.read_csv(file_path, header=None)
                if df.shape[1] < 3:
                    print(f"Skipping {file_path}: insufficient columns for metadata and spectral data.")
                    continue
                
                first_col_numeric = is_number(df.iloc[0, 0])
                second_col_numeric = is_number(df.iloc[0, 1])
                if first_col_numeric and second_col_numeric:
                    # If first two columns are numeric, assume no per-row metadata.
                    material, color = extract_info_from_filename(file_path.name)
                    colors_data = pd.Series([str(color).strip().capitalize()] * len(df))
                    materials_data = pd.Series([material] * len(df))
                    # Convert entire DataFrame to numeric.
                    df = df.apply(pd.to_numeric, errors='coerce')
                    spectral_df = df.dropna()
                else:
                    # Assume first two columns contain per-row metadata.
                    colors_data = df.iloc[:, 0].astype(str).str.strip().str.capitalize()
                    materials_data = df.iloc[:, 1].astype(str).str.strip()
                    spectral_df = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
                    # Drop rows with NaN in spectral data.
                    valid_idx = spectral_df.dropna().index
                    spectral_df = spectral_df.loc[valid_idx]
                    colors_data = colors_data.loc[valid_idx]
                    materials_data = materials_data.loc[valid_idx]
                    if spectral_df.empty:
                        print(f"Skipping {file_path} (after cleaning): no valid spectral data.")
                        continue
            
            # Convert the cleaned spectral data to a numpy array.
            spectral_data = spectral_df.values
            all_spectra.append(spectral_data)
            n_rows = spectral_data.shape[0]
            # Ensure that metadata arrays match the number of valid rows.
            colors_list = list(colors_data[:n_rows])
            materials_list = list(materials_data[:n_rows])
            # Filter out rows with missing metadata.
            valid_meta = [i for i, (c, m) in enumerate(zip(colors_list, materials_list))
                          if c and m and c.lower() != 'nan']
            colors_list = [colors_list[i] for i in valid_meta]
            materials_list = [materials_list[i] for i in valid_meta]
            # If after metadata filtering the row count changes, filter spectral_data accordingly.
            if len(valid_meta) != n_rows:
                spectral_data = spectral_data[valid_meta]
            all_colors.extend(colors_list)
            all_materials.extend(materials_list)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # If no valid data was collected, return empty arrays.
    if not all_spectra:
        return np.empty((0, 0)), np.array([]), np.array([])
    spectra_array = np.vstack(all_spectra)
    return spectra_array, np.array(all_materials), np.array(all_colors, dtype=str)

# -------------------------
# Main Training and Analysis Pipeline
# -------------------------
def main():
    base_dir = os.path.dirname(__file__)
    # Dataset configuration for each environment.
    datasets = {
        "initial": {
            "data_dir": os.path.join(base_dir, 'data_old'),
            "files": [],
            "weight": 1.0,
            "has_header": False
        },
        "current": {
            "data_dir": os.path.join(base_dir, 'data_current'),
            "files": [],
            "weight": 10.0,
            "has_header": False
        }
    }
    
    # Load the data for each environment.
    env_data = {}
    for env in datasets:
        cfg = datasets[env]
        spectra, materials, colors = load_environment_data(env, cfg["data_dir"], cfg["files"], has_header=cfg.get("has_header", False))
        env_data[env] = {"spectra": spectra, "materials": materials, "colors": colors, "weight": cfg["weight"]}
        print(f"Loaded {len(spectra)} samples for environment: {env}")
    
    # Build a label encoder on all material labels.
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
        # Skip invalid colors.
        if col.lower() == 'nan' or col.strip() == "":
            print(f"Skipping color '{col}' because it is invalid.")
            continue
        
        print(f"\n=== Training and Analysis for Color: {col} ===")
        X_env_list = []
        y_env_list = []
        env_labels_list = []  # For debugging later on
        for env in env_data:
            idx = np.where(env_data[env]["colors"] == col)[0]
            X_env = env_data[env]["spectra"][idx]
            y_env = env_data[env]["materials"][idx]
            if X_env.shape[0] > 0:
                X_env_list.append((env, X_env))
                y_env_list.append((env, y_env))
                env_labels_list.extend([env] * X_env.shape[0])
            else:
                print(f"No samples found for color {col} in environment {env}")
        if not X_env_list:
            continue
        
        # Combine raw spectral data from all environments for this color.
        X_combined_raw = np.vstack([x for (_, x) in X_env_list]).astype(float)
        if X_combined_raw.shape[0] < 2:
            print(f"Not enough samples for color {col}; skipping.")
            continue
        scaler = StandardScaler()
        X_combined_scaled = scaler.fit_transform(X_combined_raw)
        
        # Combine labels.
        y_combined_labels = np.concatenate([y for (_, y) in y_env_list])
        y_combined_encoded = material_encoder.transform(y_combined_labels)
        y_combined_final = y_combined_encoded
        
        # Optionally perform CORAL alignment between "initial" and "current" if both exist.
        X_scaled_by_env = {}
        start = 0
        for env, X_env in X_env_list:
            n_samples = X_env.shape[0]
            X_scaled_by_env[env] = X_combined_scaled[start:start + n_samples]
            start += n_samples
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
        
        # Use a Random Forest to compute feature importances.
        rf_original = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf_original.fit(X_combined_aligned, y_combined_final, sample_weight=sample_weights)
        importances = rf_original.feature_importances_
        num_features = X_combined_aligned.shape[1]
        feature_names = [f"Value{i+1}" for i in range(num_features)]
        sorted_idx = np.argsort(importances)[::-1]
        print("Feature importances (original feature space):")
        for idx in sorted_idx:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
            
        weights = compute_feature_weights(importances)
        print("Computed feature weight vector for color", col, ":")
        for i, w in enumerate(weights):
            print(f"  {feature_names[i]}: {w:.4f}")
        
        joblib.dump(weights, f'./models_per_color/{col}_feature_weights.pkl')

        
        # Compute weighted features and reduce dimensions using PCA.
        X_weighted = X_combined_aligned * weights
        scaler_weighted = StandardScaler()
        X_weighted_scaled = scaler_weighted.fit_transform(X_weighted)
        pca_weighted = PCA(n_components=2)
        try:
            X_weighted_pca = pca_weighted.fit_transform(X_weighted_scaled)
        except ValueError as e:
            print(f"PCA failed for color {col} with error: {e}. Skipping this color.")
            continue
        
        # Stratified train-test split.
        indices = np.arange(len(y_combined_final))
        train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y_combined_final)
        
        # (Optional) For headless deployment you might skip plotting and simply log metrics.
        env_labels = np.array(env_labels_list)
        train_envs, train_counts = np.unique(env_labels[train_idx], return_counts=True)
        test_envs, test_counts = np.unique(env_labels[test_idx], return_counts=True)
        print(f"Overall Training set breakdown for color {col}: {dict(zip(train_envs, train_counts))}")
        print(f"Overall Test set breakdown for color {col}: {dict(zip(test_envs, test_counts))}")
        
        X_train_w = X_weighted_pca[train_idx]
        X_test_w = X_weighted_pca[test_idx]
        y_train_w = y_combined_final[train_idx]
        y_test_w = y_combined_final[test_idx]
        w_train_w = sample_weights[train_idx]
        w_test_w = sample_weights[test_idx]
        
        unique_train_classes = np.unique(y_train_w)
        if unique_train_classes.size < 2:
            print(f"Warning: Training set for color {col} has only one class. Skipping SVM training for this color.")
            continue
        
        # Train SVM.
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
        print("Training the SVM model on weighted features...")
        svm_model.fit(X_train_w, y_train_w, sample_weight=w_train_w)
        y_train_pred_svm = svm_model.predict(X_train_w)
        y_test_pred_svm = svm_model.predict(X_test_w)
        
        # Prepare target names for present classes.
        unique_train_sorted = sorted(unique_train_classes)
        target_names_train = [material_encoder.classes_[i] for i in unique_train_sorted]
        unique_test_sorted = sorted(np.unique(y_test_w))
        target_names_test = [material_encoder.classes_[i] for i in unique_test_sorted]
        
        print("SVM Training Performance (weighted features, PCA):")
        try:
            print(classification_report(y_train_w, y_train_pred_svm,
                                        labels=unique_train_sorted,
                                        target_names=target_names_train))
        except ValueError as e:
            print("Skipping training classification report:", e)
        print("SVM Test Performance (weighted features, PCA):")
        try:
            print(classification_report(y_test_w, y_test_pred_svm,
                                        labels=unique_test_sorted,
                                        target_names=target_names_test))
        except ValueError as e:
            print("Skipping test classification report:", e)
        
        # Plot confusion matrix.
        ConfusionMatrixDisplay.from_predictions(
            y_test_w, y_test_pred_svm,
            display_labels=target_names_test,
            cmap='Oranges',
            labels=unique_test_sorted
        )
        plt.title(f"SVM: Test Confusion Matrix for Color {col}")
        plt.show()
        
          # -------------------------------
        # 3. Decision Boundary Plotting (current only)
        # -------------------------------
        # Create a meshgrid for decision boundary
        x_min, x_max = X_train_w[:, 0].min() - 1, X_train_w[:, 0].max() + 1
        y_min, y_max = X_train_w[:, 1].min() - 1, X_train_w[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = svm_model.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

      # Retrieve environment labels for the combined dataset
        env_labels = np.array(env_labels_list)
        # Extract training and test environment labels using the train/test indices.
        env_train = env_labels[train_idx]
        env_test  = env_labels[test_idx]
 
        # For training data, separate by environment.
        mask_train_current = (env_train == 'current')
        if np.any(mask_train_current):
            plt.scatter(
                X_train_w[mask_train_current, 0],
                X_train_w[mask_train_current, 1],
                c=y_train_w[mask_train_current],
                cmap=plt.cm.coolwarm,
                edgecolor='k', marker='o', s=50,
                label='Train – current'
            )
 
        # For test data, separate by environment.
        mask_test_current = (env_test == 'current')
        if np.any(mask_test_current):
            plt.scatter(
                X_test_w[mask_test_current, 0],
                X_test_w[mask_test_current, 1],
                c=y_test_w[mask_test_current],
                cmap=plt.cm.coolwarm,
                edgecolor='k', marker='^', s=80,
                label='Test – current'
            )
        # Now plot ONLY the current‐environment points (both train & test)
        env_labels = np.array(env_labels_list)
        # combine train & test masks
        train_mask = (env_labels[train_idx] == 'current')
        test_mask  = (env_labels[test_idx]  == 'current')
        if np.any(train_mask):
            plt.scatter(
                X_train_w[train_mask, 0],
                X_train_w[train_mask, 1],
                c=y_train_w[train_mask],
                cmap=plt.cm.coolwarm,
                edgecolor='k', marker='o', s=50,
                label='Train – current'
            )
        if np.any(test_mask):
            plt.scatter(
                X_test_w[test_mask, 0],
                X_test_w[test_mask, 1],
                c=y_test_w[test_mask],
                cmap=plt.cm.coolwarm,
                edgecolor='k', marker='^', s=80,
                label='Test – current'
            )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"SVM Decision Boundaries (Weighted) for Color {col}")
        plt.legend()
        plt.colorbar(label="Predicted Material Class")
        plt.show()


        
        # (Optional) Learning curve plotting (could be disabled in production)
        svm_for_curve = SVC(kernel='rbf', C=0.5, gamma='scale', class_weight='balanced', random_state=42)
        train_sizes, train_scores, val_scores = learning_curve(
            svm_for_curve, X_weighted_pca, y_combined_final, cv=5, scoring='accuracy',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, random_state=42, error_score=np.nan
        )
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Accuracy')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Accuracy')
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.title(f"SVM Learning Curve (Weighted) for Color {col}")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Save model components
        model_components = {
            'scaler': scaler,
            'scaler_weighted': scaler_weighted,
            'pca_weighted': pca_weighted,
            'svm_model': svm_model,
            'material_encoder': material_encoder,
            'rf_original': rf_original,
            'feature_weights': weights
        }
        models_per_color[col] = model_components
        joblib.dump(scaler, f'./models_per_color/{col}_scaler.pkl')
        joblib.dump(scaler_weighted, f'./models_per_color/{col}_scaler_weighted.pkl')
        joblib.dump(pca_weighted, f'./models_per_color/{col}_pca_weighted.pkl')
        joblib.dump(svm_model, f'./models_per_color/{col}_svm_model.pkl')
        print(f"Model components for color {col} saved.")
    
if __name__ == "__main__":
    main()
