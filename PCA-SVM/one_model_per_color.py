import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix

# --------
# LOAD SPECTRAL DATA FROM CSV DATA
# --------
data_dir = os.path.join(os.path.dirname(__file__), 'data')

selected_files = [
    "PLAGfulldatabase.csv", "ASAGfulldatabase.csv", "PETGfulldatabase.csv",
    "PLAKfulldatabase.csv", "ASAKfulldatabase.csv", "PETKfulldatabase.csv",
    "PLABfulldatabase.csv", "ASABfulldatabase.csv", "PETBfulldatabase.csv",
    "PLAWfulldatabase.csv", "ASAWfulldatabase.csv", "PETWfulldatabase.csv",
    "PLARfulldatabase.csv", "ASARfulldatabase.csv", "PETRfulldatabase.csv"
]

all_spectra = []
all_materials = []
all_colors = []

def extract_info_from_filename(filename):
    material = filename[:3]
    color = filename[3]
    return material, color

for filename in selected_files:
    file_path = os.path.join(data_dir, filename)
    if os.path.exists(file_path):
        material, color = extract_info_from_filename(filename)
        df = pd.read_csv(file_path, header=None)
        spectra = df.values
        all_spectra.append(spectra)
        all_materials.extend([material] * len(spectra))
        all_colors.extend([color] * len(spectra))
    else:
        print(f"Warning: File {file_path} not found. Skipping.")    

# Convert to NumPy arrays
spectra = np.vstack(all_spectra)

# Drop Channels 1, 3, 5, and 6 (index 0, 2, 4, 5)
#channels_to_remove = [0, 1, 2, 3, 4, 5]
#spectra = np.delete(spectra, channels_to_remove, axis=1)


materials = np.array(all_materials)
colors = np.array(all_colors)

# Encode material labels (we still have three classes: ASA, PET, PLA)
material_encoder = LabelEncoder()
encoded_materials = material_encoder.fit_transform(materials)

print(f"Loaded {len(spectra)} samples from {len(selected_files)} files.")
print(f"Unique materials: {np.unique(materials)} (Encoded: {np.unique(encoded_materials)})")
print(f"Unique colors: {np.unique(colors)}")

# Create a directory to save per-color models
os.makedirs('./models_per_color', exist_ok=True)

# For each unique color, train a separate model using only the spectral data.
unique_colors = np.unique(colors)
models_per_color = {}  # To store the models, scalers, and PCA components for each color

for col in unique_colors:
    print(f"\n=== Training model for Color: {col} ===")
    # Filter data for this specific color.
    idx = np.where(colors == col)[0]
    X_color = spectra[idx]   # Using only the spectra (color is constant for this subset)
    y_color = encoded_materials[idx]
    
    # Split the data into train/test (or train/val/test as needed)
    X_train, X_test, y_train, y_test = train_test_split(
        X_color, y_color, test_size=0.3, random_state=42, stratify=y_color
    )
    
    # Scale the spectral features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # (Optional) Apply PCA for visualization and possibly reducing noise.
    # You can adjust n_components based on the variance you wish to retain.
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train the SVM model on the PCA-transformed data (or directly on scaled data if you prefer)
    model = SVC(kernel='rbf', C=.5, gamma='scale', class_weight='balanced', random_state=42)
    print("Training the SVM model...")
    model.fit(X_train_pca, y_train)
    
    # Evaluate the model
    y_train_pred = model.predict(X_train_pca)
    y_test_pred = model.predict(X_test_pca)
    print("Training Performance:")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred, target_names=material_encoder.classes_))
    
    print("Test Performance:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, target_names=material_encoder.classes_))
    
    # (Optional) Display a confusion matrix for the test set
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, display_labels=material_encoder.classes_, cmap='Oranges')
    plt.title(f"Test Confusion Matrix for Color {col}")
    plt.show()
    
    # Save the scaler, PCA, and model for this color
    model_components = {
        'scaler': scaler,
        'pca': pca,
        'model': model,
        'material_encoder': material_encoder  # same encoder for all
    }
    models_per_color[col] = model_components
    
    # Save the components to disk (e.g., as models_per_color/{color}_svm_model.pkl)
    joblib.dump(scaler, f'./models_per_color/{col}_scaler.pkl')
    joblib.dump(pca, f'./models_per_color/{col}_pca.pkl')
    joblib.dump(model, f'./models_per_color/{col}_svm_model.pkl')
    joblib.dump(material_encoder, f'./models_per_color/{col}_material_encoder.pkl')
    print(f"Model components for color {col} saved.\n")
    
# Now you have separate models for each color stored in `models_per_color`
    # --------------------------
    # Visualize decision boundaries for this color model
    # --------------------------
    # Determine the grid boundaries based on the PCA-transformed training data.
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

    # Create a mesh grid over the PCA space.
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Predict over the grid using the trained model for the current color.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundaries.
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot training data points.
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                c=y_train, cmap=plt.cm.coolwarm, edgecolor='k',
                label='Train', marker='o', s=50)
    # Plot test data points.
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                c=y_test, cmap=plt.cm.coolwarm, edgecolor='k',
                label='Test', marker='^', s=80)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Decision Boundaries for Color {col}")
    plt.legend()
    plt.colorbar(label="Predicted Material Class")
    plt.show()


from sklearn.model_selection import learning_curve

# Loop over each unique color to generate a learning curve.
for col in unique_colors:
    print(f"Generating learning curve for Color: {col}")
    # Filter the data for this specific color.
    idx = np.where(colors == col)[0]
    X_color = spectra[idx]
    y_color = encoded_materials[idx]
    
    # Scale the spectral features.
    scaler = StandardScaler()
    X_color_scaled = scaler.fit_transform(X_color)
    
    # (Optional) Apply PCA if you want to work in a reduced dimension space.
    pca = PCA(n_components=2)
    X_color_pca = pca.fit_transform(X_color_scaled)
    
    # Define the SVM model to be used in the learning curve.
    svc = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=42)
    
    # Compute learning curve using 5-fold cross-validation.
    train_sizes, train_scores, val_scores = learning_curve(
        svc, X_color_pca, y_color, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, random_state=42
    )
    
    # Calculate mean accuracy for training and validation.
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    
    # Plot the learning curve.
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Accuracy')
    plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation Accuracy')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve for Color {col}")
    plt.legend()
    plt.grid(True)
    plt.show()
