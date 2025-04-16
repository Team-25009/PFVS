import os
import pandas as pd

def convert_legacy_file(input_filepath, output_filepath):
    """
    Reads a legacy CSV with a header row, identifies and removes the 'Scan' column
    (if present), identifies the color and material columns, converts spectral columns
    to numeric, and writes a new file in the following order (without a header):
      - Column 1: color (string)
      - Column 2: material (string)
      - Columns 3+: spectral values (floats)
    """
    try:
        # Read the legacy file including header row.
        df = pd.read_csv(input_filepath, header=0)
    except Exception as e:
        print(f"Error reading file {input_filepath}: {e}")
        return

    # Convert headers to lowercase for matching
    cols_lower = [col.lower() for col in df.columns]

    # Remove the 'scan' column if present.
    if 'scan' in cols_lower:
        scan_index = cols_lower.index('scan')
        # Drop using the original column name.
        df = df.drop(columns=[df.columns[scan_index]])
        # Also update the lowercased column names.
        cols_lower.pop(scan_index)

    # Identify color and material columns using known header names.
    color_col = None
    material_col = None
    for orig_col, col_lower in zip(df.columns, cols_lower):
        if col_lower.strip() == 'color':
            color_col = orig_col
        elif col_lower.strip() in ['material', 'actual material']:
            material_col = orig_col

    # If we couldn't find these columns, fall back to the first two non-numeric columns.
    if color_col is None or material_col is None:
        non_numeric_cols = df.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) >= 2:
            color_col = non_numeric_cols[0]
            material_col = non_numeric_cols[1]
        else:
            print(f"Could not determine color/material columns in {input_filepath}")
            return

    # Determine candidate spectral columns by excluding identified metadata columns.
    candidate_spec_cols = [col for col in df.columns if col not in [color_col, material_col]]
    
    # Convert candidate spectral columns to numeric; if conversion fails, they become NaN.
    spec_df = df[candidate_spec_cols].apply(pd.to_numeric, errors='coerce')
    # Drop any spectral column that is completely NaN.
    spec_df = spec_df.dropna(axis=1, how='all')

    # Drop rows that contain any NaN values in the spectral data.
    valid_rows = spec_df.notna().all(axis=1)
    if valid_rows.sum() == 0:
        print(f"No valid spectral data found in {input_filepath}")
        return

    df = df[valid_rows]
    spec_df = spec_df[valid_rows]

    # Construct a new DataFrame in the new format:
    #   - First column: color (string, trimmed)
    #   - Second column: material (string, trimmed)
    #   - Subsequent columns: spectral data (numeric)
    new_df = pd.DataFrame()
    new_df['color'] = df[color_col].astype(str).str.strip()
    new_df['material'] = df[material_col].astype(str).str.strip()

    # Rename spectral columns to a standardized format (Value1, Value2, …)
    new_spec_cols = [f"Value{i+1}" for i in range(spec_df.shape[1])]
    spec_df.columns = new_spec_cols

    # Combine metadata and spectral data.
    new_df = pd.concat([new_df.reset_index(drop=True), spec_df.reset_index(drop=True)], axis=1)

    try:
        # Save the new file without a header row.
        new_df.to_csv(output_filepath, index=False, header=False)
        print(f"Converted legacy file '{os.path.basename(input_filepath)}' to new format and saved as '{output_filepath}'.")
    except Exception as e:
        print(f"Error writing file {output_filepath}: {e}")

def convert_legacy_folder(input_folder, output_folder):
    """
    Processes all CSV files in the input_folder by converting each to the new format.
    Converted files are saved to the output_folder with the same filename.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".csv")]
    for file_name in files:
        in_path = os.path.join(input_folder, file_name)
        out_path = os.path.join(output_folder, file_name)
        convert_legacy_file(in_path, out_path)

if __name__ == "__main__":
    # Define the folder paths (adjust paths as needed).
    legacy_folder = "data_old"
    converted_folder = "data_old_converted"
    
    print("Starting conversion of legacy files...")
    convert_legacy_folder(legacy_folder, converted_folder)
    print("Conversion complete.")
