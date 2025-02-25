import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
import os

# --------------------------
# Step 1: Load and Prepare Data
# --------------------------
def load_data(file_path):
    """
    Load spectral data from CSV file.
    """
    df = pd.read_csv(file_path)
    spectra = df.iloc[:, :-3].values.tolist()  # Convert to list to remove NumPy dependency
    materials = df['Material'].tolist()    # Predict only 'Material'
    colors = df['Color'].tolist()          # Use 'Color' as an input feature
    return spectra, materials, colors, df

# Load real dataset
file_path = os.path.join(os.path.dirname(__file__), 'data', 'DecemberScans.csv')
spectra, materials, colors, df = load_data(file_path)

# Encode material labels and colors
material_encoder = LabelEncoder()
encoded_materials = material_encoder.fit_transform(materials).tolist()

color_encoder = LabelEncoder()
encoded_colors = color_encoder.fit_transform(colors).tolist()

# Combine spectral data with color information
combined_features = [spectra[i] + [encoded_colors[i]] for i in range(len(spectra))]

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features).tolist()

# --------------------------
# Step 2: Train-Test-Validation Split
# --------------------------
X_train, X_temp, y_train, y_temp, colors_train, colors_temp = train_test_split(
    scaled_features, encoded_materials, colors, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test, colors_val, colors_test = train_test_split(
    X_temp, y_temp, colors_temp, test_size=0.5, random_state=42
)

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
os.makedirs('./models', exist_ok=True)
joblib.dump(scaler, './models/scaler.pkl')
joblib.dump(model, './models/svm_model.pkl')
joblib.dump(material_encoder, './models/material_encoder.pkl')
joblib.dump(color_encoder, './models/color_encoder.pkl')
print("\nModel and scaler saved in 'models' folder.")

# --------------------------
# Step 6: Visualize with PCA
# --------------------------
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_features)

plt.scatter([x[0] for x in reduced_data], [x[1] for x in reduced_data], c=y_train, cmap='viridis', alpha=0.7)
plt.title("PCA Visualization of Spectral Data with Color Information")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Material Type")
plt.show()

# --------------------------
# Step 7: Load Model for Predictions
# --------------------------
def predict_new_sample(new_sample, color_label):
    """
    Predict material type for a new sample.
    """
    loaded_scaler = joblib.load('./models/scaler.pkl')
    loaded_model = joblib.load('./models/svm_model.pkl')
    loaded_material_encoder = joblib.load('./models/material_encoder.pkl')
    loaded_color_encoder = joblib.load('./models/color_encoder.pkl')

    encoded_color = loaded_color_encoder.transform([color_label])[0]
    combined_sample = new_sample + [encoded_color]
    scaled_sample = loaded_scaler.transform([combined_sample]).tolist()

    prediction = loaded_model.predict(scaled_sample)
    return loaded_material_encoder.inverse_transform(prediction)[0]

# Example prediction
new_sample = [0.5] * 18  # Replace with real spectral data
predicted_material = predict_new_sample(new_sample, 'Red')  # Example color label
print("\nPredicted Material:", predicted_material)
