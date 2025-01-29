import os
import numpy as np
import pandas as pd

def generate_dirty_spectral_data(n_samples=3000, n_bins=18, n_classes=3):
    """
    Generate synthetic spectral data for testing with added noise and variability.
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
        # Generate a base spectrum for this class
        base_spectrum = np.random.uniform(0.5, 1.5, size=n_bins)
        
        for _ in range(n_samples // n_classes):
            # Add noise to the base spectrum
            noise = np.random.normal(0, 0.1, size=n_bins)  # Small noise
            if np.random.rand() < 0.1:  # 10% chance of large noise (outlier)
                noise += np.random.normal(0, 0.5, size=n_bins)
            
            # Simulate slight drift in base spectrum
            drift = base_spectrum + np.random.normal(0, 0.05, size=n_bins)
            
            # Combine drift and noise
            noisy_spectrum = drift + noise
            
            # Randomly overlap some features with other classes (simulate shared features)
            if np.random.rand() < 0.2:  # 20% chance of overlapping features
                overlap = np.random.choice(range(n_bins), size=3, replace=False)
                for i in overlap:
                    noisy_spectrum[i] = base_spectrum[i] + np.random.uniform(-0.1, 0.1)
            
            spectra.append(noisy_spectrum)
            labels.append(f"Material_{class_id + 1}")
    
    # Add completely random outlier samples (1% of total samples)
    for _ in range(int(n_samples * 0.01)):
        random_sample = np.random.uniform(-1, 2, size=n_bins)  # Random values in a larger range
        spectra.append(random_sample)
        labels.append(f"Material_{np.random.choice(range(1, n_classes + 1))}")
    
    return np.array(spectra), np.array(labels)

# Generate data
spectra, labels = generate_dirty_spectral_data()

# Create the output folder if it doesn't exist
data_folder = "../data/"
os.makedirs(data_folder, exist_ok=True)

# Save the generated data to the data/ folder
spectra_file = os.path.join(data_folder, "spectral_data.csv")
labels_file = os.path.join(data_folder, "labels.csv")

pd.DataFrame(spectra).to_csv(spectra_file, index=False)
pd.DataFrame(labels, columns=["Label"]).to_csv(labels_file, index=False)

print(f"Dirty test data generated and saved to:\n  {spectra_file}\n  {labels_file}")
