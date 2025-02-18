import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    """
    Load spectral data from CSV file.
    """
    df = pd.read_csv(file_path)
    spectra = df.iloc[:, :-3].values  # Adjusted to exclude the new 'Color' column
    materials = df['Material'].values    # Predict only 'Material'
    colors = df['Color'].values          # Use 'Color' as an input feature
    return spectra, materials, colors, df

# Load real dataset
file_path = os.path.join(os.path.dirname(__file__), 'data', 'ALLRLNG.csv')
spectra, materials, colors, df = load_data(file_path)

# Encode material labels and colors
material_encoder = LabelEncoder()
encoded_materials = material_encoder.fit_transform(materials)

color_encoder = LabelEncoder()
encoded_colors = color_encoder.fit_transform(colors)

# Combine spectral data with color information
combined_features = np.column_stack((spectra, encoded_colors))

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

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
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) #rpf

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

# Save individual predictions to CSV with mismatch highlighting
print("\nSaving Detailed Test Predictions to CSV...")
test_results = pd.DataFrame({
    'Actual Material': material_encoder.inverse_transform(y_test),
    'Color': colors_test,  # Use original color labels directly
    'Predicted Material': material_encoder.inverse_transform(y_test_pred)
})

# Highlight mismatches
test_results['Mismatch'] = test_results['Actual Material'] != test_results['Predicted Material']

# Generate overall statistics
color_stats = test_results.groupby('Color')['Mismatch'].value_counts().unstack(fill_value=0)
material_stats = test_results.groupby('Actual Material')['Mismatch'].value_counts().unstack(fill_value=0)

# Print statistics to terminal
print("\nColor Stats:")
print(color_stats)

print("\nMaterial Stats:")
print(material_stats)

# Save statistics to a text file
results_folder = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_folder, exist_ok=True)

stats_file = os.path.join(results_folder, os.path.basename(file_path).replace('.csv', '_stats.txt'))
with open(stats_file, 'w') as f:
    f.write("Color Stats:\n")
    f.write(color_stats.to_string())
    f.write("\n\nMaterial Stats:\n")
    f.write(material_stats.to_string())

print(f"Statistics saved to {stats_file}")

# Generate output file name
output_file = file_path.replace('.csv', '_TestConfirmation.csv')
test_results.to_csv(output_file, index=False)

print(f"Detailed predictions saved to {output_file}")

# --------------------------
# Step 5: Save Model and Scaler
# --------------------------
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

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=encoded_materials, cmap='viridis', alpha=0.7)
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
    combined_sample = np.append(new_sample, encoded_color).reshape(1, -1)
    scaled_sample = loaded_scaler.transform(combined_sample)

    prediction = loaded_model.predict(scaled_sample)
    return loaded_material_encoder.inverse_transform(prediction)[0]

# Example prediction
new_sample = np.random.rand(1, 18)  # Replace with real spectral data
predicted_material = predict_new_sample(new_sample, 'Red')  # Example color label
print("\nPredicted Material:", predicted_material)