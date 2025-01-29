import os
import glob
import numpy as np

def process_files(input_folder, output_file):
    """
    Processes all text files in the given folder. Each file is assumed to be a CSV-like structure with
    rows of numeric values. The program aggregates all rows across all files, computes some statistics (e.g. averages),
    and writes out a reference file containing the average values.

    Parameters:
        input_folder (str): The directory containing the input .txt or .csv files.
        output_file (str): The path to an output CSV file that will store results.

    Returns:
        None
    """
    # Find all text files in the folder (adjust the pattern if your files have different extensions)
    file_paths = glob.glob(os.path.join(input_folder, '*.txt')) + glob.glob(os.path.join(input_folder, '*.csv'))
    
    if not file_paths:
        print("No files found in directory:", input_folder)
        return

    # This will store all scans from all files
    # We will end up with a list of numpy arrays, one per line (scan).
    all_scans = []

    # Iterate over each file
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # skip empty lines
                    continue
                # Split by comma and convert each field to float
                values = line.split(',')
                try:
                    scan = np.array([float(v) for v in values], dtype=np.float64)
                except ValueError:
                    # If there's a header or non-numeric line, you might need to skip it
                    print(f"Skipping non-numeric line in file {file_path}: {line}")
                    continue
                all_scans.append(scan)

    if not all_scans:
        print("No valid scan data found.")
        return

    # Convert list of arrays into a single 2D array: rows = scans, columns = spectral values
    all_scans_array = np.vstack(all_scans)

    # Compute averages column-wise. This gives you an average spectrum.
    # Adjust as needed: you could, for example, compute averages per file or per group.
    average_spectrum = np.mean(all_scans_array, axis=0)

    # Write the average to the output file
    # If you have multiple groupings or want more complex stats, you can do that too.
    with open(output_file, 'w') as out_f:
        out_line = ','.join([f"{val:.4f}" for val in average_spectrum])
        out_f.write(out_line + '\n')
    
    print("Processing complete.")
    print("Average spectrum saved to:", output_file)


if __name__ == "__main__":
    # Example usage:
    # Place all your text/csv files in a folder named "data"
    # Then run:
    # python process_scans.py
    #
    # This will produce a file named "averaged_spectrum.csv" containing the averaged values.
    
    input_folder = r"C:\Users\malco\Desktop\Senior Design\PFVS\data"
    output_file = "averaged_spectrum.csv"
    process_files(input_folder, output_file)
