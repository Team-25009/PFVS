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

# Reading Data from csv
            # this section is building up code of what the load_data(file_path) function will do
            # anytime this fuction is used, it'll complete the following lines

def load_data(file_path):
    """
    Load spectral data from CSV file.
    """
    df = pd.read_csv(file_path, header=0)
    spectra = df.iloc[:, :-2].values  # Adjusted to exclude the new 'Color' column
    materials = df['Material'].values    # Predict only 'Material'
    colors = df['Color'].values          # Use 'Color' as an input feature
    return spectra, materials, colors, df

            # ---
            # reads the csv file (file_path variable, designated below)
            # extracts the spectral data (spectra var.) for all columns except last two
            # ## df allows us to read from the csv file anytime
            # mat & colors are reading from those columns, extrating mat labels and color labels


# Load real dataset
file_path = os.path.join(os.path.dirname(__file__), 'data', 'ALLRLNG.csv')
spectra, materials, colors, df = load_data(file_path)
            
            # ---
            # filepath - creates path to the dataset/ csv file
            # uses the function created above to create seperated dataset


# Encode material labels and colors
material_encoder = LabelEncoder()
encoded_materials = material_encoder.fit_transform(materials)

color_encoder = LabelEncoder()
encoded_colors = color_encoder.fit_transform(colors)
            
            #---
            # encodes material and color labels into a numerical form
            # ex: pla = 1, asa = 2, petg = 3, red = 1, blue = 1, etc


# Combine spectral data with color information
combined_features = np.column_stack((spectra, encoded_colors))

            #---
            # adds encoded color info to the spectral data

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

            #standardizes features (mean=0, variance =1) to improve model performance

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
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) 

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
