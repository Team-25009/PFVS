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
def load_data():
    """
    Replace this function with actual data loading.
    Returns:
        spectra (numpy array): Feature matrix [n_samples, n_features].
        labels (numpy array): Corresponding labels [n_samples].
    """
    # Example: Simulated random data
    spectra = np.random.rand(3000, 18)  # Replace with actual spectral data
    labels = np.random.choice(["PLA", "PETG", "ASA"], size=3000)  # Replace with actual labels
    return spectra, labels

# Load data
spectra, labels = load_data()

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Scale features
scaler = StandardScaler()
scaled_spectra = scaler.fit_transform(spectra)

# --------------------------
# Step 2: Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    scaled_spectra, encoded_labels, test_size=0.2, random_state=42
)

# --------------------------
# Step 3: Initialize and Train SVM Model
# --------------------------
# Initialize the model
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the model
print("Training the SVM model...")
model.fit(X_train, y_train)

# --------------------------
# Step 4: Evaluate the Model
# --------------------------
# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# Step 5: Hyperparameter Tuning (Optional)
# --------------------------
def tune_model(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("\nBest Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Uncomment to perform hyperparameter tuning
# model = tune_model(X_train, y_train)

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
# Step 7: Save the Model for Deployment
# --------------------------
# Save the scaler and model for later use on Raspberry Pi
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'svm_model.pkl')

print("\nModel and scaler saved as 'svm_model.pkl' and 'scaler.pkl'.")

# --------------------------
# Step 8: Loading the Model for Prediction
# --------------------------
def predict_new_sample(new_sample):
    """
    Predict material type for a new sample.
    Args:
        new_sample (numpy array): Single sample of spectral data [1, n_features].
    Returns:
        str: Predicted material type.
    """
    # Load saved scaler and model
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_model = joblib.load('svm_model.pkl')
    
    # Preprocess and predict
    scaled_sample = loaded_scaler.transform(new_sample)
    prediction = loaded_model.predict(scaled_sample)
    return label_encoder.inverse_transform(prediction)[0]

# Example: Predicting on a new sample
new_sample = np.random.rand(1, 18)  # Replace with actual spectral data
predicted_material = predict_new_sample(new_sample)
print("\nPredicted Material:", predicted_material)
