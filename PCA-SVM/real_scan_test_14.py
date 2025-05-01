import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def load_model_components(color, models_dir):
    """
    Load the scaler, PCA, SVM model, and label encoder for a given color.
    """
    scaler = joblib.load(os.path.join(models_dir, f"{color}_scaler.pkl"))
    pca = joblib.load(os.path.join(models_dir, f"{color}_pca.pkl"))
    model = joblib.load(os.path.join(models_dir, f"{color}_svm_model.pkl"))
    encoder = joblib.load(os.path.join(models_dir, f"{color}_material_encoder.pkl"))
    return scaler, pca, model, encoder

def load_training_subset(color_abbrev, data_dir, materials=["PLA", "ASA", "PET"], n_scans=50):
    """
    Load first n_scans from each material database for a given color abbreviation.
    Removes Channels 1, 3, 5, and 6.
    """
    subset_rows = []
    for material in materials:
        filename = f"{material}{color_abbrev}fulldatabase.csv"
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            df_mat = pd.read_csv(file_path, header=None)
            df_mat = df_mat.iloc[:n_scans, :].copy()
            df_mat.columns = [f"Value{i}" for i in range(1, df_mat.shape[1]+1)]
            df_mat.insert(0, "Scan", range(1, len(df_mat)+1))
            df_mat.insert(0, "Material", material)
            df_mat.insert(0, "Color", color_abbrev)
            subset_rows.append(df_mat)
        else:
            print(f"Warning: {file_path} not found.")
    if subset_rows:
        return pd.concat(subset_rows, ignore_index=True)
    else:
        return pd.DataFrame()

def plot_decision_boundary(model, X_pca, title, zoom_factor=2.0, 
                           X_test_pca=None, test_labels=None,
                           X_train_pca=None, train_labels=None):
    """
    Plot the decision boundary (in PCA space) with optional overlays.
    """
    all_points = X_pca.copy()
    if X_test_pca is not None:
        all_points = np.vstack([all_points, X_test_pca])
    if X_train_pca is not None:
        all_points = np.vstack([all_points, X_train_pca])
    
    x_min_orig = all_points[:, 0].min() - 1
    x_max_orig = all_points[:, 0].max() + 1
    y_min_orig = all_points[:, 1].min() - 1
    y_max_orig = all_points[:, 1].max() + 1

    x_center = (x_min_orig + x_max_orig) / 2
    y_center = (y_min_orig + y_max_orig) / 2
    x_half_range_zoom = ((x_max_orig - x_min_orig) / 2) * zoom_factor
    y_half_range_zoom = ((y_max_orig - y_min_orig) / 2) * zoom_factor
    x_min = x_center - x_half_range_zoom
    x_max = x_center + x_half_range_zoom
    y_min = y_center - y_half_range_zoom
    y_max = y_center + y_half_range_zoom

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    if X_test_pca is not None:
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                    c='black', edgecolor='w', marker='o', s=100,
                    label="Real Scans")
    if X_train_pca is not None:
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                    c='yellow', edgecolor='k', marker='s', s=100,
                    label="Training Subset")

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.colorbar(label="Predicted Class (Encoded)")
    plt.show()

def main():
    # Setup paths
    script_dir = os.path.dirname(__file__)
    test_file = os.path.join(script_dir, "data", "real_scans.txt")
    data_dir = os.path.join(script_dir, "data")
    models_dir = os.path.join(script_dir, "models_per_color")

    # Load real scan data
    df_real = pd.read_csv(test_file)
    print("Real scans file read successfully!")
    print(df_real.head())

    # Define spectral columns and channels to remove
    spectral_columns = [f"Value{i}" for i in range(1, 19)]
    channels_to_remove = [0, 1, 2, 3, 4, 5]  # Corresponds to Channels 1, 3, 5, 6

    # Map full color names to abbreviations used in model training
    color_map = {
        "Red": "R",
        "Green": "G",
        "Blue": "B",
        "Black": "K",
        "White": "W"
    }

    all_predictions = []

    for full_color in df_real["Color"].unique():
        print(f"\nProcessing color: {full_color}")
        color_abbrev = color_map.get(full_color, full_color)

        df_real_color = df_real[df_real["Color"] == full_color].copy()
        X_real = df_real_color[spectral_columns].values.astype(float)
        X_real = np.delete(X_real, channels_to_remove, axis=1)

        # Load trained model components
        try:
            scaler, pca, model, encoder = load_model_components(color_abbrev, models_dir)
        except FileNotFoundError as e:
            print(f"Error loading models for color {color_abbrev}: {e}")
            continue

        # Apply model pipeline to real scans
        X_real_scaled = scaler.transform(X_real)
        X_real_pca = pca.transform(X_real_scaled)

        # Predict materials
        pred_encoded_real = model.predict(X_real_pca)
        pred_material_real = encoder.inverse_transform(pred_encoded_real)
        df_real_color["Predicted_Material"] = pred_material_real
        print(df_real_color[["Color", "Scan", "Predicted_Material"]])
        all_predictions.append(df_real_color)

        # Load and process training subset for overlay
        df_train_subset = load_training_subset(color_abbrev, data_dir, n_scans=50)
        if df_train_subset.empty:
            print(f"No training data found for color {color_abbrev}.")
            X_train_pca = None
        else:
            X_train = df_train_subset[[col for col in df_train_subset.columns if col.startswith("Value")]].values.astype(float)
            X_train = np.delete(X_train, channels_to_remove, axis=1)
            X_train_scaled = scaler.transform(X_train)
            X_train_pca = pca.transform(X_train_scaled)

        # Plot decision boundary
        title = f"Decision Boundary for Color {full_color} (Model: {color_abbrev})"
        plot_decision_boundary(model, X_real_pca, title, zoom_factor=2.0,
                               X_test_pca=X_real_pca,
                               X_train_pca=X_train_pca)

    # Save predictions
    if all_predictions:
        df_all = pd.concat(all_predictions, ignore_index=True)
        output_file = os.path.join(script_dir, "predicted_test_scans.csv")
        df_all.to_csv(output_file, index=False)
        print(f"\nAll real scan predictions have been saved to {output_file}")

if __name__ == '__main__':
    main()
