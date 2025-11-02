import pandas as pd
import numpy as np
from pathlib import Path

def compute_differences(folder_path, norm_filename='norm.csv'):
    """
    Compute differences between norm.csv and all only*.csv files in a folder.
    
    Args:
        folder_path: Path to folder containing the CSV files
        norm_filename: Name of the norm/baseline CSV file (default: 'norm.csv')
    """
    folder = Path(folder_path)
    
    # Load the norm data
    norm_path = folder / norm_filename
    if not norm_path.exists():
        print(f"Error: {norm_filename} not found in {folder_path}")
        return
    
    norm_data = pd.read_csv(norm_path)
    print(f"Loaded norm data from {norm_filename}")
    
    # Find all abnorm_only*.csv files
    abnorm_files = sorted(folder.glob('only*.csv'))
    
    if not abnorm_files:
        print(f"only*.csv files found in {folder_path}")
        return
    
    print(f"Found {len(abnorm_files)} only file(s)\n")
    
    # Process each abnorm file
    for abnorm_file in abnorm_files:
        print(f"Processing: {abnorm_file.name}")
        
        # Extract suffix (e.g., "0" from "abnorm_only0.csv")
        # Remove "abnorm_only" prefix and ".csv" suffix
        suffix = abnorm_file.stem.replace('only', '')
        
        # Load OOD data
        ood_data = pd.read_csv(abnorm_file)
        
        # Compute difference (Norm - OOD)
        difference = pd.DataFrame({
            'timestep': norm_data['timestep'],
            'mean_diff': norm_data['mean'] - ood_data['mean'],
        })
        
        # Optional: Add uncertainty propagation for std
        if 'std' in norm_data.columns and 'std' in ood_data.columns:
            # Standard error propagation for difference
            difference['std_diff'] = np.sqrt(norm_data['std']**2 + ood_data['std']**2)
            difference['mean_diff_plus_std'] = difference['mean_diff'] + difference['std_diff']
            difference['mean_diff_minus_std'] = difference['mean_diff'] - difference['std_diff']
        
        # Find the 3 largest differences (most positive)
        largest_3 = difference.nlargest(3, 'mean_diff')
        smallest = difference.nsmallest(1, 'mean_diff')
        
        # Create dictionary with extrema timesteps
        extrema = {
            '1stMax': int(largest_3.iloc[0]['timestep']),
            '2ndMax': int(largest_3.iloc[1]['timestep']),
            '3rdMax': int(largest_3.iloc[2]['timestep']),
            'min': int(smallest.iloc[0]['timestep'])
        }
        
        # Create output filename: difference_{suffix}s.csv
        output_filename = f'difference_{suffix}s.csv'
        output_path = folder / output_filename
        
        # Save difference data
        difference.to_csv(output_path, index=False)
        
        print(f"  ✓ Saved to: {output_filename}")
        print(f"  Extrema timesteps: {extrema}")
        print(f"  Preview:")
        print(difference.head().to_string(index=False))
        print("-" * 60 + "\n")
    
    print(f"Finished processing {len(abnorm_files)} file(s)")

if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute differences between norm.csv and only*.csv files'
    )
    parser.add_argument('folder', help='Folder containing CSV files')
    parser.add_argument('--norm', default='norm.csv', help='Name of norm CSV file (default: norm.csv)')
    
    args = parser.parse_args()
    
    compute_differences(args.folder, args.norm)