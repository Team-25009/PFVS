#--- importing libraries

import os   # handles file paths
import pandas as pd   # loading and processing csv data
import numpy as np   # support for numerical operations
import joblib   # saving and loading machine learning models
from sklearn.preprocessing import LabelEncoder, StandardScaler   # encodes catergorical labels (ex: 'redpla') into a numerical format
from sklearn.model_selection import train_test_split  # splits dataset into train, val, test datasets (we have set %'s for this) 
from sklearn.svm import SVC  # implements support vector classification (SVC is a ML tecnique, hyperplane to sep. data into groups)
from sklearn.decomposition import PCA  #  principal component analysis (PCA, preprocessing technique, find largest variations (principal components) )
import matplotlib.pyplot as plt  # plots and visualizes data
from sklearn.metrics import accuracy_score, classification_report  # generates the classification performance summary at end

# --------
# LOAD SPECTRAL DATA FROM CSV DATA
# --------

# Define the directory containing CSV files
data_dir = os.path.join(os.path.dirname(__file__), 'data')
            
            #---
            # refers to the data subfolder, and constructs full path


# List of specific files to process
selected_files = ["PLARLNG.csv", "ASARLNG.csv", "PETRLNG.csv"]  # Update as needed

            #---
            # only list the files here that we want
            # allows us to quickly review by color, mat type, etc for different algorithm tests
            # any file not listed will not be included (so we can have multiple files in data folder)

# Initialize lists to store all data
all_spectra = []
all_materials = []
all_colors = []

            #---
            # stores all spectra data, material labels, colors from all the selected files


# Function to extract material and color from filename
def extract_info_from_filename(filename):
    material = filename[:3]  # First 3 characters (e.g., 'PLA')
    color = filename[3]      # 4th character (e.g., 'R' for red)
    return material, color

            #---
            # SETS UP A DEFINED FUNCTION
            # Extracts the material & color info from the file names! :D
            # first 3 characters = filament type
                # PLA 
                # ASA
                # PET = PETG
            # 4th character = color type
                # R = Red
                # B = Blue
                # G = Green
                # W = White
                # K = Black



# Read only the specified CSV files
for filename in selected_files:
    file_path = os.path.join(data_dir, filename)  # Corrected path
            
            #---
            # Loops through all of your selected files, creates full file path

    if os.path.exists(file_path):  # Ensure the file exists
        # Extract material and color from filename
        material, color = extract_info_from_filename(filename)

        # Load spectral data (assuming no headers in the file)
        df = pd.read_csv(file_path, header=None)
        spectra = df.values  # Convert to NumPy array

        # Store data
        all_spectra.append(spectra)
        all_materials.extend([material] * len(spectra))  # Repeat material label
        all_colors.extend([color] * len(spectra))  # Repeat color label

            #--- steps
                # 1: checks file exists
                # 2: calls definied function 'extract_info_from_filename', allows us to get the material name and color
                # 3: reads the csv file ( we don't have any headers )
                # 4: merges everything into a single file
                    # all spectra - spectra file
                    # repeats the material label for each row in file
                    # repeats color label for each row in file
                # ensures each row of data is paired with correct material and color

    else:
        print(f"Warning: File {file_path} not found. Skipping.")



# Convert lists to NumPy arrays
spectra = np.vstack(all_spectra)  # Stack all spectral data
materials = np.array(all_materials)
colors = np.array(all_colors)

            #---
            # stacks spectral data into a single NumPy array
            # converts materials and colors into a NumPy array 

# Encode materials and colors
material_encoder = LabelEncoder()
encoded_materials = material_encoder.fit_transform(materials)

color_encoder = LabelEncoder()
encoded_colors = color_encoder.fit_transform(colors)

            #---
            # encodes material and color labels into a numerical form
            # ex: pla = 0, asa = 1, petg = 2, red = 0, blue = 1, etc
            # The specific encoded labels will be printed at end


# Combine spectral data with both material and color information
combined_features = np.column_stack((spectra, encoded_materials, encoded_colors))

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
pca = PCA(n_components=2)  # Adjust the number of components if needed
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
X_train, X_temp, y_train, y_temp, colors_train, colors_temp = train_test_split(
    pca_features, encoded_materials, colors, test_size=0.3, random_state=42
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

# Example prediction
new_sample = np.random.rand(1, 18)  # Replace with real spectral data
predicted_material = predict_new_sample(new_sample, 'Red')  # Example color label
print("\nPredicted Material:", predicted_material)
