import os
import numpy as np
import pandas as pd

def generate_spectral_data(n_samples=3000, n_bins=18, n_classes=3):
    """
    Generate synthetic spectral data for testing.
    Args:
        n_samples (int): Total number of samples.
        n_bins (int): Number of spectral bins (features).
        n_classes (int): Number of material types.
    Returns:
        spectra (numpy array): Feature matrix [n_samples, n_bins].
        labels (numpy array): Corresponding labels [n_samples].
    """
    np.random.seed(42)
    spectra = []
    labels = []
    for class_id in range(n_classes):
        # Generate a distinct "signature" for each material
        base_spectrum = np.random.uniform(0.5, 1.5, size=n_bins)
        for _ in range(n_samples // n_classes):
            # Add random noise to the base spectrum
            noisy_spectrum = base_spectrum + np.random.normal(0, 0.1, size=n_bins)
            spectra.append(noisy_spectrum)
            labels.append(f"Material_{class_id + 1}")
    
    return np.array(spectra), np.array(labels)

# Generate data
spectra, labels = generate_spectral_data()

# Create the output folder if it doesn't exist
data_folder = "../data/"
os.makedirs(data_folder, exist_ok=True)

# Save the generated data to the data/ folder
spectra_file = os.path.join(data_folder, "spectral_data.csv")
labels_file = os.path.join(data_folder, "labels.csv")

pd.DataFrame(spectra).to_csv(spectra_file, index=False)
pd.DataFrame(labels, columns=["Label"]).to_csv(labels_file, index=False)

print(f"Test data generated and saved to:\n  {spectra_file}\n  {labels_file}")
