#--- importing libraries

import os   # handles file paths
import pandas as pd   # loading and processing csv data
import numpy as np   # support for numerical operations
import joblib   # saving and loading machine learning models
from sklearn.preprocessing import LabelEncoder, StandardScaler   # encodes categorical labels into a numerical format
from sklearn.model_selection import train_test_split  # splits dataset into train, val, test datasets
from sklearn.svm import SVC  # implements support vector classification
from sklearn.decomposition import PCA  # principal component analysis
import matplotlib.pyplot as plt  # plots and visualizes data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold

# --------
# LOAD SPECTRAL DATA FROM CSV DATA
# --------

data_dir = os.path.join(os.path.dirname(__file__), 'data')
selected_files = ["PLARLNG.csv", "ASARLNG.csv", "PETRLNG.csv"]

all_spectra = []
all_materials = []
all_colors = []

def extract_info_from_filename(filename):
    material = filename[:3]  # First 3 characters (e.g., 'PLA')
    color = filename[3]      # 4th character (e.g., 'R' for red)
    return material, color

for filename in selected_files:
    file_path = os.path.join(data_dir, filename)
    if os.path.exists(file_path):
        material, color = extract_info_from_filename(filename)
        df = pd.read_csv(file_path, header=None)
        spectra = df.values  # Convert to NumPy array
        all_spectra.append(spectra)
        all_materials.extend([material] * len(spectra))
        all_colors.extend([color] * len(spectra))
    else:
        print(f"Warning: File {file_path} not found. Skipping.")

# Convert lists to NumPy arrays
spectra = np.vstack(all_spectra)
materials = np.array(all_materials)
colors = np.array(all_colors)

# Encode materials and colors
material_encoder = LabelEncoder()
encoded_materials = material_encoder.fit_transform(materials)

color_encoder = LabelEncoder()
encoded_colors = color_encoder.fit_transform(colors)

# Combine spectral data with only color information (NOT material)
combined_features = np.column_stack((spectra, encoded_colors))

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

# Display dataset info
print(f"Loaded {len(spectra)} samples from {len(selected_files)} files.")
print(f"Unique materials: {np.unique(materials)} (Encoded: {np.unique(encoded_materials)})")
print(f"Unique colors: {np.unique(colors)} (Encoded: {np.unique(encoded_colors)})")

# --------------------------
# Step 1: PCA
# --------------------------
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.scatter(pca_features[:, 0], pca_features[:, 1], c=encoded_materials, cmap='viridis', alpha=0.7)
plt.title("PCA Visualization of Spectral Data with Color Information")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Material Type")
plt.show()

# --------------------------
# Step 2: Train-Test-Validation Split
# --------------------------
X_train, X_temp, y_train, y_temp = train_test_split(pca_features, encoded_materials, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --------------------------
# Step 3: Initialize and Train SVM Model
# --------------------------
model = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42) 

print("Training the SVM model...")
model.fit(X_train, y_train)

# --------------------------
# Step 4: Evaluate the Model
# --------------------------
y_val_pred = model.predict(X_val)
print("\nValidation Performance:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred, target_names=material_encoder.classes_))

y_test_pred = model.predict(X_test)
print("\nTest Performance:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=material_encoder.classes_))

# --------------------------
# Step 5: Save Model and Scaler
# --------------------------
joblib.dump(scaler, './models/scaler.pkl')
joblib.dump(pca, './models/pca.pkl')
joblib.dump(model, './models/svm_model.pkl')
joblib.dump(material_encoder, './models/material_encoder.pkl')
joblib.dump(color_encoder, './models/color_encoder.pkl')
print("\nModel, PCA, and scaler saved in 'models' folder.")

# --------------------------
# Step 6: Load Model for Predictions
# --------------------------
def predict_new_sample(new_sample, color_label):
    """
    Predict material type for a new sample.
    """
    loaded_scaler = joblib.load('./models/scaler.pkl')
    loaded_pca = joblib.load('./models/pca.pkl')
    loaded_model = joblib.load('./models/svm_model.pkl')
    loaded_material_encoder = joblib.load('./models/material_encoder.pkl')
    loaded_color_encoder = joblib.load('./models/color_encoder.pkl')

    encoded_color = loaded_color_encoder.transform([color_label])[0]
    combined_sample = np.append(new_sample, encoded_color).reshape(1, -1)
    scaled_sample = loaded_scaler.transform(combined_sample)
    pca_sample = loaded_pca.transform(scaled_sample)

    prediction = loaded_model.predict(pca_sample)
    return loaded_material_encoder.inverse_transform(prediction)[0]

# --------------------------
# Step 7: K-Fold Cross-Validation
# --------------------------
k = 5  # Number of folds
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

cross_val_scores = cross_val_score(model, pca_features, encoded_materials, cv=cv, scoring='accuracy')
print(f"\n{k}-Fold Cross-Validation Accuracy Scores: {cross_val_scores}")
print(f"Mean Accuracy: {np.mean(cross_val_scores):.4f}")
print(f"Standard Deviation: {np.std(cross_val_scores):.4f}")

# --------------------------
# Step 8: Confusion Matrix
# --------------------------
y_test_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=material_encoder.classes_)
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Confusion Matrix for SVM Model")
plt.show()
