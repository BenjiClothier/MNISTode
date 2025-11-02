import numpy as np
import pandas as pd
import argparse
import os
import glob

def process_results_for_tikz(npy_file, output_csv, compute_std=True, compute_ci=True):
    """
    Process results with shape (num_samples, num_timesteps) for tikz plotting.
    Computes mean, std, and confidence intervals across samples.
    
    Args:
        npy_file: path to .npy file with shape (samples, timesteps)
        output_csv: path to output CSV
        compute_std: whether to compute standard deviation
        compute_ci: whether to compute 95% confidence intervals
    """
    # Load data
    data = np.load(npy_file)

    if data[1].shape == 1:
        data = data.squeeze(1)
    print(f"Loaded data with shape: {data.shape}")
    print(f"  {data.shape[0]} samples × {data.shape[1]} timesteps")
    
    num_samples, num_timesteps = data.shape
    timesteps = np.arange(num_timesteps)
    
    # Compute statistics across samples (axis=0)
    mean = np.mean(data, axis=0)
    
    results = {
        'timestep': timesteps,
        'mean': mean
    }
    
    if compute_std:
        std = np.std(data, axis=0)
        results['std'] = std
        results['mean_plus_std'] = mean + std
        results['mean_minus_std'] = mean - std
    
    if compute_ci:
        # 95% confidence interval
        sem = np.std(data, axis=0) / np.sqrt(num_samples)  # Standard error of mean
        ci = 1.96 * sem  # 95% CI
        results['ci_upper'] = mean + ci
        results['ci_lower'] = mean - ci
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to: {output_csv}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nSummary statistics:")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    if compute_std:
        print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    return df


def process_directory(directory, output_dir=None, compute_std=True, compute_ci=True):
    """
    Process all only*.npy files in a directory.
    
    Args:
        directory: path to directory containing only*.npy files
        output_dir: output directory (if None, uses same as input directory)
        compute_std: whether to compute standard deviation
        compute_ci: whether to compute 95% confidence intervals
    """
    if output_dir is None:
        output_dir = directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all only*.npy files
    pattern = os.path.join(directory, "only*.npy")
    npy_files = glob.glob(pattern)
    
    if not npy_files:
        print(f"No 'only*.npy' files found in {directory}")
        return
    
    print(f"Found {len(npy_files)} file(s) matching 'only*.npy'")
    print("=" * 60)
    
    processed_files = []
    
    for npy_file in sorted(npy_files):
        filename = os.path.basename(npy_file)
        print(f"\nProcessing: {filename}")
        print("-" * 60)
        
        # Generate output filename (only0.npy -> only0.csv)
        base_name = filename.replace('.npy', '.csv')
        output_csv = os.path.join(output_dir, base_name)
        
        try:
            process_results_for_tikz(npy_file, output_csv, compute_std, compute_ci)
            processed_files.append((filename, base_name))
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Processing complete! Processed {len(processed_files)} file(s):")
    for input_name, output_name in processed_files:
        print(f"  {input_name} -> {output_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Process results for tikz plotting (mean across samples)'
    )
    
    # Make input and output mutually exclusive with directory mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('input', nargs='?', help='Input .npy file with shape (samples, timesteps)')
    group.add_argument('-d', '--directory', help='Process all only*.npy files in directory')
    
    parser.add_argument('-o', '--output', help='Output .csv file (for single file mode) or output directory (for directory mode)')
    parser.add_argument('--no-std', action='store_true', help='Skip standard deviation')
    parser.add_argument('--no-ci', action='store_true', help='Skip confidence intervals')
    
    args = parser.parse_args()
    
    if args.directory:
        # Directory mode: process all only*.npy files
        process_directory(
            args.directory,
            output_dir=args.output,
            compute_std=not args.no_std,
            compute_ci=not args.no_ci
        )
    else:
        # Single file mode
        if not args.output:
            parser.error("the following arguments are required: -o/--output")
        
        process_results_for_tikz(
            args.input,
            args.output,
            compute_std=not args.no_std,
            compute_ci=not args.no_ci
        )


if __name__ == '__main__':
    main()