import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

# --------------------------
# Step 1: Load and Prepare Data
# --------------------------
def load_data(file_path):
    """
    Load spectral data from CSV file.
    """
    df = pd.read_csv(file_path)
    spectra = df.iloc[:, :-2].values  # Use first 18 columns as features
    labels = df['Material'].values    # Use 'Material' column as labels
    return spectra, labels

# Load real dataset
file_path = r"C:\Users\malco\Desktop\Senior Design\PFVS\SVM_Project\data\DecemberScans.csv"  # Update path if necessary
spectra, labels = load_data(file_path)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Scale features
scaler = StandardScaler()
scaled_spectra = scaler.fit_transform(spectra)

# --------------------------
# Step 2: Train-Test-Validation Split
# --------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    scaled_spectra, encoded_labels, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# --------------------------
# Step 3: Initialize and Train SVM Model
# --------------------------
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

print("Training the SVM model...")
model.fit(X_train, y_train)

# --------------------------
# Step 4: Evaluate the Model
# --------------------------
y_val_pred = model.predict(X_val)

print("\nValidation Performance:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

y_test_pred = model.predict(X_test)

print("\nTest Performance:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# --------------------------
# Step 5: Save Model and Scaler
# --------------------------
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'svm_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("\nModel and scaler saved as 'svm_model.pkl' and 'scaler.pkl'.")

# --------------------------
# Step 6: Visualize with PCA
# --------------------------
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_spectra)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=encoded_labels, cmap='viridis', alpha=0.7)
plt.title("PCA Visualization of Spectral Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Material Type")
plt.show()

# --------------------------
# Step 7: Load Model for Predictions
# --------------------------
def predict_new_sample(new_sample):
    """
    Predict material type for a new sample.
    """
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_model = joblib.load('svm_model.pkl')
    loaded_label_encoder = joblib.load('label_encoder.pkl')
    
    scaled_sample = loaded_scaler.transform(new_sample)
    prediction = loaded_model.predict(scaled_sample)
    return loaded_label_encoder.inverse_transform(prediction)[0]

# Example prediction
new_sample = np.random.rand(1, 18)  # Replace with real spectral data
predicted_material = predict_new_sample(new_sample)
print("\nPredicted Material:", predicted_material)
