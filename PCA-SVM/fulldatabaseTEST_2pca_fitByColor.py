#--- importing libraries

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
from sklearn.model_selection import StratifiedShuffleSplit, learning_curve

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
materials = np.array(all_materials)
colors = np.array(all_colors)

# Encode labels
material_encoder = LabelEncoder()
encoded_materials = material_encoder.fit_transform(materials)

color_encoder = LabelEncoder()
encoded_colors = color_encoder.fit_transform(colors)

# Combine spectral data with encoded color ONLY (material is the target)
combined_features = np.column_stack((spectra, encoded_colors))

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

# Info
print(f"Loaded {len(spectra)} samples from {len(selected_files)} files.")
print(f"Unique materials: {np.unique(materials)} (Encoded: {np.unique(encoded_materials)})")
print(f"Unique colors: {np.unique(colors)} (Encoded: {np.unique(encoded_colors)})")

# --------------------------
# Step 1: PCA (for both visualization and training)
# --------------------------
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.scatter(pca_features[:, 0], pca_features[:, 1], c=encoded_materials, cmap='viridis', alpha=0.7)
plt.title("PCA Visualization of Spectral Data with Color Info")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Material Type")
plt.show()

# --------------------------
# Step 2: Stratified Train-Test Split (use scaled features, not PCA)
# --------------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in sss.split(scaled_features, encoded_materials):
    X_train = scaled_features[train_idx]
    y_train = encoded_materials[train_idx]
    color_train = colors[train_idx]

    X_temp = scaled_features[temp_idx]
    y_temp = encoded_materials[temp_idx]
    color_temp = colors[temp_idx]

X_val, X_test, y_val, y_test, color_val, color_test = train_test_split(
    X_temp, y_temp, color_temp, stratify=y_temp, test_size=0.5, random_state=42
)

# --------------------------
# Step 3: Apply PCA to all sets for training
# --------------------------
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# --------------------------
# Step 4: Train SVM Model on PCA-transformed features
# --------------------------
model = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=42)
print("Training the SVM model on PCA-transformed features...")
model.fit(X_train_pca, y_train)

# --------------------------
# Step 5: Evaluate Model
# --------------------------
y_train_pred = model.predict(X_train_pca)
print("\nTraining Performance:")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred, target_names=material_encoder.classes_))

y_val_pred = model.predict(X_val_pca)
print("\nValidation Performance:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred, target_names=material_encoder.classes_))

y_test_pred = model.predict(X_test_pca)
print("\nTest Performance:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, target_names=material_encoder.classes_))

# Confusion matrices
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, display_labels=material_encoder.classes_, cmap='Blues')
plt.title("Training Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, display_labels=material_encoder.classes_, cmap='Greens')
plt.title("Validation Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, display_labels=material_encoder.classes_, cmap='Oranges')
plt.title("Test Confusion Matrix")
plt.show()

# Per-color accuracy and confusion matrices
color_groups = {}
for i, color in enumerate(color_test):
    if color not in color_groups:
        color_groups[color] = {'y_true': [], 'y_pred': []}
    color_groups[color]['y_true'].append(y_test[i])
    color_groups[color]['y_pred'].append(y_test_pred[i])

print("\nPer-Color Accuracy and Confusion Matrices:")
for color, results in color_groups.items():
    y_true = results['y_true']
    y_pred = results['y_pred']
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{color}: {accuracy:.2f}")
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=material_encoder.classes_).plot(cmap='Purples')
    plt.title(f"Confusion Matrix for Color {color}")
    plt.show()

# Learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_pca, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', label='Training score')
plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation score')
plt.title("Learning Curve (SVM with 2 PCA Components)")
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# --------------------------
# Step 6: Save Components
# --------------------------
os.makedirs('./models', exist_ok=True)
joblib.dump(scaler, './models/scaler.pkl')
joblib.dump(pca, './models/pca.pkl')
joblib.dump(model, './models/svm_model.pkl')
joblib.dump(material_encoder, './models/material_encoder.pkl')
joblib.dump(color_encoder, './models/color_encoder.pkl')
print("\nModel, PCA, and scaler saved in 'models' folder.")

# --------------------------
# Step 7: Prediction Function
# --------------------------
def predict_new_sample(new_sample, color_label):
    loaded_scaler = joblib.load('./models/scaler.pkl')
    loaded_pca = joblib.load('./models/pca.pkl')
    loaded_model = joblib.load('./models/svm_model.pkl')
    material_encoder = joblib.load('./models/material_encoder.pkl')
    color_encoder = joblib.load('./models/color_encoder.pkl')

    encoded_color = color_encoder.transform([color_label])[0]
    combined_sample = np.append(new_sample, encoded_color).reshape(1, -1)
    scaled_sample = loaded_scaler.transform(combined_sample)
    pca_sample = loaded_pca.transform(scaled_sample)

    prediction = loaded_model.predict(pca_sample)
    return material_encoder.inverse_transform(prediction)[0]

