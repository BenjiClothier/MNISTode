import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path

def npy_to_csv(npy_file, csv_file=None, column_names=None, index_label=None):
    """
    Convert a .npy file to .csv format.
    
    Args:
        npy_file: path to input .npy file
        csv_file: path to output .csv file (if None, uses same name with .csv extension)
        column_names: list of column names (optional)
        index_label: label for the index column (optional)
    """
    # Load the numpy array
    data = np.load(npy_file)
    
    print(f"Loaded array with shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Handle different array shapes
    if data.ndim == 1:
        # 1D array - convert to column vector
        df = pd.DataFrame(data, columns=column_names or ['value'])
    elif data.ndim == 2:
        # 2D array - each column is a series
        if column_names and len(column_names) != data.shape[1]:
            print(f"Warning: Expected {data.shape[1]} column names, got {len(column_names)}")
            column_names = None
        df = pd.DataFrame(data, columns=column_names)
    else:
        # Higher dimensional - flatten to 2D
        print(f"Warning: {data.ndim}D array detected. Reshaping to 2D...")
        data_2d = data.reshape(data.shape[0], -1)
        df = pd.DataFrame(data_2d)
    
    # Set output filename
    if csv_file is None:
        csv_file = Path(npy_file).with_suffix('.csv')
    
    # Save to CSV
    df.to_csv(csv_file, index=True, index_label=index_label or 'index')
    print(f"Saved to: {csv_file}")
    print(f"CSV shape: {df.shape}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Convert .npy files to .csv for LaTeX tikz/pgfplots')
    parser.add_argument('input', help='Input .npy file')
    parser.add_argument('-o', '--output', help='Output .csv file (default: same name as input)')
    parser.add_argument('-c', '--columns', nargs='+', help='Column names (space-separated)')
    parser.add_argument('-i', '--index-label', default='index', help='Index column label (default: "index")')
    parser.add_argument('--no-index', action='store_true', help='Do not write index column')
    
    args = parser.parse_args()
    
    # Load and convert
    data = np.load(args.input)
    print(f"Loaded array with shape: {data.shape}")
    
    # Handle different array shapes
    if data.ndim == 1:
        df = pd.DataFrame(data, columns=args.columns or ['value'])
    elif data.ndim == 2:
        df = pd.DataFrame(data, columns=args.columns)
    else:
        print(f"Warning: {data.ndim}D array detected. Reshaping to 2D...")
        data_2d = data.reshape(data.shape[0], -1)
        df = pd.DataFrame(data_2d)
    
    # Set output filename
    output = args.output or Path(args.input).with_suffix('.csv')
    
    # Save to CSV
    df.to_csv(output, index=not args.no_index, index_label=args.index_label if not args.no_index else None)
    print(f"✓ Saved to: {output}")
    print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")


if __name__ == '__main__':
    main()