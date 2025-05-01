import os
import pandas as pd

def extract_info_from_filename(filename):
    """
    Extracts the material and color from the filename.
    Assumes that the first three characters represent the material and the fourth character represents the color.
    Example: "ASABfulldatabase.csv" produces material = "ASA" and color = "B".
    """
    base = os.path.basename(filename)
    material = base[:3]
    color = base[3]
    return material, color

def convert_old_to_new_format(input_dir, output_dir):
    """
    Converts all CSV files from the old database format (in input_dir) into the new format,
    and saves them in output_dir.
    
    New format:
      - Header: Color, Material, Scan, Value1, Value2, ...
      - 'Scan' is an enumeration for each row.
    """
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop over every CSV file in input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            try:
                # Read the old file (assumed to have no header)
                df = pd.read_csv(file_path, header=None)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            
            # Extract metadata from the filename
            material, color = extract_info_from_filename(filename)
            
            # Rename spectral columns to Value1, Value2, ...
            num_values = df.shape[1]
            df.columns = [f"Value{i+1}" for i in range(num_values)]
            
            # Insert the "Scan", "Material", and "Color" columns in that order
            # Inserting reversely so that final order is: Color, Material, Scan, Value1, Value2, ...
            df.insert(0, "Scan", range(1, df.shape[0] + 1))
            df.insert(0, "Material", material)
            df.insert(0, "Color", color)
            
            # Define output file path (preserve same filename)
            output_file = os.path.join(output_dir, filename)
            try:
                df.to_csv(output_file, index=False)
                print(f"Converted {filename} and saved to {output_file}")
            except Exception as e:
                print(f"Error saving {output_file}: {e}")

def main():
    # Set the input folder (old database) and the output folder (new format)
    base_dir = os.path.dirname(__file__)
    input_folder = os.path.join(base_dir, "data")      # Folder with old-format CSV files
    output_folder = os.path.join(base_dir, "data_old")   # New folder to store the converted files

    convert_old_to_new_format(input_folder, output_folder)

if __name__ == "__main__":
    main()
