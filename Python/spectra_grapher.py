import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_sensor_data(file_path, wavelength_start=410, wavelength_end=960):
    """
    Plots the intensity vs. wavelength for multiple sensor runs and the average.

    Args:
        file_path (str): Path to the .txt file containing the dataset.
        wavelength_start (float): Starting wavelength (e.g., 410 nm).
        wavelength_end (float): Ending wavelength (e.g., 960 nm).
    """
    # Load the dataset
    try:
        # Read the .txt file as comma-separated values
        df = pd.read_csv(file_path, header=None, delimiter=',')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Define the wavelengths based on the number of columns
    num_columns = df.shape[1]
    wavelengths = np.linspace(wavelength_start, wavelength_end, num_columns)

    # Compute the average intensity for each column
    average_intensities = df.mean(axis=0)

    # Plot all sensor runs
    plt.figure(figsize=(14, 8))
    for row in df.itertuples(index=False):
        plt.plot(wavelengths, row, alpha=0.3, color='gray', linewidth=0.8)

    # Overlay the average intensity
    plt.plot(wavelengths, average_intensities, color='red', label='Average Intensity', linewidth=2)

    # Add labels and title
    plt.title("Sensor Intensity Across Runs")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

# Example usage
# Save your dataset as a .txt file and provide its path below
file_path = r"C:\Users\malco\Downloads\RedPLA200.txt"
 # Replace with your .txt file path
plot_sensor_data(file_path)
