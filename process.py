import numpy as np
import pandas as pd
import argparse

def process_results_for_tikz(npy_file, output_csv, compute_std=True, compute_ci=True, save_raw=True):
    """
    Process results with shape (num_samples, num_timesteps) for tikz plotting.
    Computes mean, std, and confidence intervals across samples.
    
    Args:
        npy_file: path to .npy file with shape (samples, timesteps)
        output_csv: path to output CSV
        compute_std: whether to compute standard deviation
        compute_ci: whether to compute 95% confidence intervals
        save_raw: whether to save raw data for 2D histogram (in long format)
    """
    # Load data
    data = np.load(npy_file)
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
    
    print(f"\n✓ Saved statistics to: {output_csv}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nSummary statistics:")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    if compute_std:
        print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    # Save raw data if requested (for 2D histogram)
    if save_raw:
        raw_output = output_csv.replace('.csv', '_raw.csv')
        
        # Convert to long format: each row is (timestep, sample_id, value)
        raw_data = []
        for sample_idx in range(num_samples):
            for timestep_idx in range(num_timesteps):
                raw_data.append({
                    'timestep': timestep_idx,
                    'sample_id': sample_idx,
                    'value': data[sample_idx, timestep_idx]
                })
        
        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv(raw_output, index=False)
        
        print(f"\n✓ Saved raw data to: {raw_output}")
        print(f"  Shape: {len(df_raw)} rows (for 2D histogram)")
        print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Process results for tikz plotting (mean across samples)'
    )
    parser.add_argument('input', help='Input .npy file with shape (samples, timesteps)')
    parser.add_argument('-o', '--output', required=True, help='Output .csv file')
    parser.add_argument('--no-std', action='store_true', help='Skip standard deviation')
    parser.add_argument('--no-ci', action='store_true', help='Skip confidence intervals')
    parser.add_argument('--save-raw', action='store_true', 
                       help='Save raw data in long format for 2D histogram')
    
    args = parser.parse_args()
    
    process_results_for_tikz(
        args.input,
        args.output,
        compute_std=not args.no_std,
        compute_ci=not args.no_ci,
        save_raw=args.save_raw
    )

if __name__ == '__main__':
    main()