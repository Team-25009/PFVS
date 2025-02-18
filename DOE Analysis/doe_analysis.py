import os
import numpy as np
import pandas as pd
from scipy import stats


def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        trials = []

        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            values = [value.strip() for value in line.split(",") if value.strip()]
            numeric_values = [float(value) for value in values]
            trials.append(numeric_values)

    return np.array(trials)


def calculate_statistics(data_array):
    mean_values = np.mean(data_array, axis=0)
    std_values = np.std(data_array, axis=0)
    range_values = np.ptp(data_array, axis=0)
    mode_values = stats.mode(data_array, axis=0, keepdims=True).mode[0]
    
    return mean_values, std_values, range_values, mode_values

def process_files_in_folder(folder_path, output_file):
    all_data = []  # List to hold data for all files
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".TXT"):
            file_path = os.path.join(folder_path, file_name)
            data_array = read_data_from_file(file_path)
            mean_values, std_values, range_values, mode_values = calculate_statistics(data_array)
            
            for i, (mean, std, value_range, mode) in enumerate(zip(mean_values, std_values, range_values, mode_values), 1):
                all_data.append({
                    "File Name": file_name,
                    "Wavelength": i,
                    "Mean": round(mean, 2),
                    "Std Dev": round(std, 2),
                    "Range": value_range,
                    "Mode": mode
                })
    
    # Create a DataFrame and save it to an Excel file
    df = pd.DataFrame(all_data)
    df.to_excel(output_file, index=False, sheet_name="Statistics")


folder_path = "./Raspberry Pi scans"  # Change this to your actual folder path
output_file = "Pi_scans.xlsx"

# Process all .TXT files in the folder and save results
process_files_in_folder(folder_path, output_file)

print(f"Processing complete. Results saved in {output_file}")