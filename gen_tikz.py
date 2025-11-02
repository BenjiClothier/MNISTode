import os
import glob
import re
import argparse

def generate_tikz_plots(directory, learned_digit='1'):
    """
    Generate LaTeX TikZ plots for each only*.csv file in the directory.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing norm.csv, only*.csv, and difference*.csv files
    learned_digit : str
        The digit/label for the learned (in-distribution) data (default: '1')
    
    Returns:
    --------
    dict : Dictionary with keys as full filenames and values as LaTeX code strings
    """
    
    # Find all only*.csv files
    only_files = glob.glob(os.path.join(directory, "only*.csv"))
    
    if not only_files:
        print(f"No 'only*.csv' files found in {directory}")
        return {}
    
    # Check if norm.csv exists
    norm_path = os.path.join(directory, "norm.csv")
    if not os.path.exists(norm_path):
        print(f"Warning: norm.csv not found in {directory}")
        return {}
    
    plots = {}
    
    for only_file in only_files:
        # Extract filename
        filename = os.path.basename(only_file)
        
        # Extract first digit after "only" for labeling
        match = re.search(r'only(\d)', filename)
        if not match:
            print(f"Warning: Could not extract digit from {filename}, skipping")
            continue
        
        digit_label = match.group(1)
        
        # Extract full identifier for difference file matching
        # (e.g., "only1_100s.csv" -> "1_100s")
        full_identifier = filename.replace("only", "").replace(".csv", "")
        
        # Build paths
        only_path = only_file
        
        # Try different possible difference file patterns
        possible_diff_files = [
            os.path.join(directory, f"difference_{full_identifier}.csv"),
            os.path.join(directory, f"difference_{full_identifier}s.csv"),
            os.path.join(directory, f"difference_{digit_label}s.csv"),
        ]
        
        diff_path = None
        for possible_path in possible_diff_files:
            if os.path.exists(possible_path):
                diff_path = possible_path
                break
        
        if diff_path is None:
            print(f"Warning: No matching difference file found for {filename}, skipping")
            print(f"  Tried: {[os.path.basename(p) for p in possible_diff_files]}")
            continue
        
        # Generate LaTeX code with configurable learned_digit
        latex_code = f"""\\begin{{figure}}[htbp]
 \\centering
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
 group style={{
 group size=1 by 2, % 1 column, 2 rows
 vertical sep=1.5cm, % Space between plots
 xlabels at=edge bottom, % Only bottom plot gets x-label
 }},
 width=0.85\\textwidth,
 height=5cm,
 grid=major,
 legend style={{
 font=\\small,
 /tikz/every even column/.append style={{column sep=0.2em}}
 }},
 ]
% ===== TOP PLOT: Original comparison =====
\\nextgroupplot[
 ylabel={{$\\log p(x)$}},
 ymode=log,
 legend pos=south east,
 title={{Likelihood Comparison: {learned_digit}'s (in dist) data vs {digit_label}'s (out dist)}},
 ]
% Mean line normal data
\\addplot[blue, thick] table[x=timestep, y=mean, col sep=comma] {{{norm_path}}};
\\addlegendentry{{Learned data \\{{{learned_digit}\\}}}}
% Mean ± std shaded region norm
\\addplot[name path=upper_norm, draw=none, forget plot]
 table[x=timestep, y=mean_plus_std, col sep=comma] {{{norm_path}}};
\\addplot[name path=lower_norm, draw=none, forget plot]
 table[x=timestep, y=mean_minus_std, col sep=comma] {{{norm_path}}};
\\addplot[blue!20, fill opacity=0.3] fill between[of=upper_norm and lower_norm];
\\addlegendentry{{Learned data std $\\pm$}}
% Mean line abnormal data
\\addplot[red, thick] table[x=timestep, y=mean, col sep=comma] {{{only_path}}};
\\addlegendentry{{OOD {digit_label}'s}}
% Mean ± std shaded region abnormal data
\\addplot[name path=upper_cats, draw=none, forget plot]
 table[x=timestep, y=mean_plus_std, col sep=comma] {{{only_path}}};
\\addplot[name path=lower_cats, draw=none, forget plot]
 table[x=timestep, y=mean_minus_std, col sep=comma] {{{only_path}}};
\\addplot[red!20, fill opacity=0.3] fill between[of=upper_cats and lower_cats];
\\addlegendentry{{OOD {digit_label}'s std $\\pm$}}
% ===== BOTTOM PLOT: Difference =====
\\nextgroupplot[
 xlabel={{Timestep}},
 ylabel={{Difference}},
 legend pos=south west,
 title={{Difference (Learned - OOD)}},
 ]
% Difference line
\\addplot[green!50!black, thick] table[x=timestep, y=mean_diff, col sep=comma] {{{diff_path}}};
\\addlegendentry{{Mean difference}}
% Optional: add std bands for difference
\\addplot[name path=upper_diff, draw=none, forget plot]
 table[x=timestep, y=mean_diff_plus_std, col sep=comma] {{{diff_path}}};
\\addplot[name path=lower_diff, draw=none, forget plot]
 table[x=timestep, y=mean_diff_minus_std, col sep=comma] {{{diff_path}}};
\\addplot[green!20, fill opacity=0.3] fill between[of=upper_diff and lower_diff];
\\addlegendentry{{Difference std $\\pm$}}
% Zero reference line
\\addplot[black, dashed, thin, forget plot] coordinates {{(0,0) (99,0)}};
\\end{{groupplot}}
\\end{{tikzpicture}}
\\caption{{Top: Comparison of learned ({learned_digit}'s) vs OOD ({digit_label}'s) data likelihoods. Bottom: Difference between the two.}}
\\end{{figure}}
"""
        
        # Use filename without extension as key
        key = filename.replace('.csv', '')
        plots[key] = latex_code
    
    return plots


def save_plots_to_files(directory, learned_digit='1', output_dir=None):
    """
    Generate and save plots to separate .tex files
    
    Parameters:
    -----------
    directory : str
        Path to directory containing the CSV files
    learned_digit : str
        The digit/label for the learned (in-distribution) data (default: '1')
    output_dir : str, optional
        Directory to save output .tex files. If None, uses the same directory.
    """
    if output_dir is None:
        output_dir = directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plots = generate_tikz_plots(directory, learned_digit=learned_digit)
    
    if not plots:
        print("No plots generated.")
        return plots
    
    print(f"\nGenerating {len(plots)} plot(s)...")
    print("=" * 60)
    
    for key, latex_code in plots.items():
        output_file = os.path.join(output_dir, f"plot_{key}.tex")
        with open(output_file, 'w') as f:
            f.write(latex_code)
        print(f"✓ Saved: {output_file}")
    
    print("=" * 60)
    print(f"Done! Generated {len(plots)} plot file(s) in {output_dir}")
    
    return plots


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX TikZ plots for OOD analysis from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots with default learned digit (1)
  python script.py /path/to/data

  # Specify learned digit as 7
  python script.py /path/to/data --learned-digit 7

  # Save to a different output directory
  python script.py /path/to/data -o /path/to/output

  # Combine options
  python script.py /path/to/data --learned-digit 3 -o ./plots
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory containing norm.csv, only*.csv, and difference*.csv files'
    )
    
    parser.add_argument(
        '-l', '--learned-digit',
        default='1',
        help='Digit/label for the learned (in-distribution) data (default: 1)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for .tex files (default: same as input directory)'
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.directory):
        parser.error(f"Directory not found: {args.directory}")
    
    # Generate and save plots
    save_plots_to_files(
        directory=args.directory,
        learned_digit=args.learned_digit,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()