import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def extract_color_code(full_tag):
    """
    Given a tag like "BlackASA" or "BluePETG", remove the material suffix
    (where material tags can be 'PETG', 'ASA', 'PET', or 'PLA') and then return
    the color code: 'K' for Black (to avoid confusion with Blue) or the first
    letter for any other color.
    """
    # Check material tags in order (longest first to handle PETG correctly)
    material_tags = ['PETG', 'ASA', 'PET', 'PLA']
    for tag in material_tags:
        if full_tag.endswith(tag):
            color_name = full_tag[:-len(tag)].strip()
            break
    else:
        color_name = full_tag.strip()
    
    if color_name.lower() == 'black':
        return 'K'
    else:
        return color_name[0].upper()

def extract_material_tag(full_tag):
    """
    Given a tag like "BlackASA" or "BluePETG", return the material part.
    Checks for 'PETG' (4 letters) first, then for ASA, PET, or PLA (3 letters).
    """
    material_tags = ['PETG', 'ASA', 'PET', 'PLA']
    for tag in material_tags:
        if full_tag.endswith(tag):
            return tag
    return full_tag.strip()

# Define the CSV path (assumes the CSV is in the same directory as the script)
csv_path = os.path.join(os.path.dirname(__file__), './data/DecemberScans.csv')

# Read the CSV file (it has a header row)
df = pd.read_csv(csv_path)

# The first 18 columns (columns '0' through '17') are spectral data.
spectra = df.loc[:, df.columns[:-2]].values

# Extract the target material from the "Material" column using our helper.
df["TargetMaterial"] = df["Material"].apply(extract_material_tag)

# Extract the color code from the "Color" column using our helper.
df["ColorCode"] = df["Color"].apply(extract_color_code)

# Encode the target material (e.g., ASA, PETG, PLA)
material_encoder = LabelEncoder()
encoded_material = material_encoder.fit_transform(df["TargetMaterial"])

print(f"Loaded {len(spectra)} samples from DecemberScans.csv")
print(f"Unique target materials: {np.unique(df['TargetMaterial'])} (Encoded: {np.unique(encoded_material)})")
print(f"Unique colors: {np.unique(df['ColorCode'])}")

# Create a directory to save per-color models
os.makedirs('./models_per_color', exist_ok=True)

# Train a separate model for each unique color based on the extracted color code.
unique_colors = np.unique(df["ColorCode"])
models_per_color = {}  # To store the models, scalers, and PCA components for each color

for col in unique_colors:
    print(f"\n=== Training model for Color: {col} ===")
    # Filter the data for this specific color.
    idx = np.where(df["ColorCode"] == col)[0]
    X_color = spectra[idx]          # Spectral data for samples with this color.
    y_color = encoded_material[idx] # Target material labels for these samples.
    
    # Split the data into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_color, y_color, test_size=0.3, random_state=42, stratify=y_color
    )
    
    # Scale the spectral features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA for dimensionality reduction or visualization.
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train an SVM model on the PCA-transformed data.
    model = SVC(kernel='rbf', C=0.5, gamma='scale', class_weight='balanced', random_state=42)
    print("Training the SVM model...")
    model.fit(X_train_pca, y_train)
    
    # Evaluate the model.
    y_train_pred = model.predict(X_train_pca)
    y_test_pred = model.predict(X_test_pca)
    print("Training Performance:")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    # Determine which classes are present in y_train and adjust target names accordingly.
    unique_train_labels = np.unique(y_train)
    print(classification_report(
        y_train, y_train_pred,
        labels=unique_train_labels,
        target_names=material_encoder.inverse_transform(unique_train_labels)
    ))
    
    print("Test Performance:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    unique_test_labels = np.unique(y_test)
    print(classification_report(
        y_test, y_test_pred,
        labels=unique_test_labels,
        target_names=material_encoder.inverse_transform(unique_test_labels)
    ))
    
    # Display the confusion matrix for the test set.
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, display_labels=material_encoder.inverse_transform(unique_test_labels), cmap='Oranges'
    )
    plt.title(f"Test Confusion Matrix for Color {col}")
    plt.show()
    
    # Save the scaler, PCA, and model for this color.
    model_components = {
        'scaler': scaler,
        'pca': pca,
        'model': model,
        'material_encoder': material_encoder  # Same encoder for all models.
    }
    models_per_color[col] = model_components
    
    # Save components to disk.
    joblib.dump(scaler, f'./models_per_color/{col}_scaler.pkl')
    joblib.dump(pca, f'./models_per_color/{col}_pca.pkl')
    joblib.dump(model, f'./models_per_color/{col}_svm_model.pkl')
    joblib.dump(material_encoder, f'./models_per_color/{col}_material_encoder.pkl')
    print(f"Model components for color {col} saved.\n")
    
    # Visualize decision boundaries for the current color model.
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot training and test data.
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                c=y_train, cmap=plt.cm.coolwarm, edgecolor='k',
                label='Train', marker='o', s=50)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                c=y_test, cmap=plt.cm.coolwarm, edgecolor='k',
                label='Test', marker='^', s=80)
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Decision Boundaries for Color {col}")
    plt.legend()
    plt.colorbar(label="Predicted Material Class")
    plt.show()

# Generate learning curves for each color model.
for col in unique_colors:
    print(f"Generating learning curve for Color: {col}")
    idx = np.where(df["ColorCode"] == col)[0]
    X_color = spectra[idx]
    y_color = encoded_material[idx]
    
    # Scale the spectral features.
    scaler = StandardScaler()
    X_color_scaled = scaler.fit_transform(X_color)
    
    # Apply PCA.
    pca = PCA(n_components=2)
    X_color_pca = pca.fit_transform(X_color_scaled)
    
    # Define the SVM model.
    svc = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=42)
    
    # Compute the learning curve using 5-fold cross-validation.
    train_sizes, train_scores, val_scores = learning_curve(
        svc, X_color_pca, y_color, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, random_state=42
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Accuracy')
    plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation Accuracy')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve for Color {col}")
    plt.legend()
    plt.grid(True)
    plt.show()
