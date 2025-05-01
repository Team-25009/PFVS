import os
import pandas as pd
from pathlib import Path

# Mapping of single-letter color codes to full color names.
color_mapping = {
    'B': 'blue',
    'R': 'red',
    'K': 'black',
    'G': 'green',
    'W': 'white'
}

def reformat_file(file_path, overwrite=False):
    """
    Reads a CSV file (assumed to have no header) and checks if the first column
    contains a single letter that is in our color mapping. If so, the script replaces
    that value with the full color name.
    
    Parameters:
      file_path (Path): the Path object pointing to the CSV file.
      overwrite (bool): if True, overwrite the file; if False, write the output with
                        a '_reformatted' suffix.
    """
    try:
        # Read the CSV file with no header (all files are assumed to be spectral data)
        df = pd.read_csv(file_path, header=None)
        
        # Check if the file has at least one row and one column.
        if df.shape[0] == 0 or df.shape[1] == 0:
            print(f"Skipping empty file: {file_path}")
            return
        
        # Check the first row, first column value.
        first_val = str(df.iloc[0, 0]).strip().upper()
        if first_val in color_mapping:
            # Replace values in column 0 that are exactly one letter and in our mapping.
            def map_color(value):
                val = str(value).strip().upper()
                return color_mapping.get(val, value)
            
            df.iloc[:, 0] = df.iloc[:, 0].apply(map_color)
            
            # Determine output file name.
            if overwrite:
                output_file = file_path
            else:
                output_file = file_path.with_name(file_path.stem + '_reformatted.csv')
            
            # Write the reformatted DataFrame back to CSV without header/index.
            df.to_csv(output_file, header=False, index=False)
            print(f"Processed {file_path} -> {output_file}")
        else:
            print(f"File {file_path} does not appear to require reformatting (first column not a known color code).")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def reformat_all_in_folder(folder_path, overwrite=False):
    """
    Scans through all CSV files in the given folder and applies reformatting.
    
    Parameters:
      folder_path (str or Path): Directory containing the files.
      overwrite (bool): Whether to overwrite files or save new reformatted files.
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return
    
    for csv_file in csv_files:
        reformat_file(csv_file, overwrite=overwrite)

if __name__ == "__main__":
    # Set your folder path for old_data
    folder_path = "data_old"  # Adjust this path as needed.
    # Set overwrite to False to save new files; set True to overwrite originals.
    reformat_all_in_folder(folder_path, overwrite=False)
