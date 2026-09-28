"""
Fit Gaussian models to quantization errors for Q-Drift correction.

This script loads FP16 and quantized outputs collected by collect_statistics.py
and fits channel-wise Gaussian models for each timestep. The joint distribution
of quantized output and quantization error is modeled per-channel as:

    [X, Y] ~ N(mu, Sigma)
    
where X is quantized output and Y is quantization error (quant_output - fp16_output).

Follows D2-DPM's gaussian_modeling.py: one model per timestep and channel, fitted
after 4-sigma outlier removal. Used for the channel-wise ablation and D2-DPM rows.
"""

import argparse
import os
from typing import Dict, Tuple
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm


def compute_log_likelihood(quant: np.ndarray, error: np.ndarray, 
                          mean: np.ndarray, covariance: np.ndarray) -> float:
    """
    Compute log-likelihood to evaluate how well data fits the Gaussian model.
    
    Args:
        quant: Quantized output values (1D array)
        error: Quantization error values (1D array)
        mean: Mean vector [mu_quant, mu_error] shape (2,)
        covariance: 2x2 covariance matrix
    
    Returns:
        float: Average log-likelihood (higher is better)
    """
    try:
        # Stack data points: each row is [quant, error]
        data_points = np.vstack((quant, error)).T  # shape (N, 2)
        
        # Multivariate normal log-likelihood
        mvn = stats.multivariate_normal(mean=mean, cov=covariance, allow_singular=True)
        log_likelihood = np.mean(mvn.logpdf(data_points))
        
        return log_likelihood
    except Exception as e:
        print(f"    Warning: Failed to compute log-likelihood: {e}")
        return np.nan


def print_modeling_summary(
    log_likelihood_dict: Dict[int, Dict[int, float]],
    mu_dict: Dict[int, np.ndarray],
    cov_dict: Dict[int, np.ndarray],
):
    """
    Print and return a summary of channel-wise Gaussian modeling quality.
    
    Args:
        log_likelihood_dict: {timestep: {channel: log_likelihood}}
        mu_dict: {timestep: (2, C) array}
        cov_dict: {timestep: (2, 2, C) array}
    
    Returns:
        tuple: (avg_ll, summary_text)
    """
    # Auto-detect number of channels from data
    first_timestep = next(iter(mu_dict.keys()))
    num_channels = int(mu_dict[first_timestep].shape[1])

    summary_lines = []
    
    summary_lines.append("="*80)
    summary_lines.append("GAUSSIAN MODELING QUALITY REPORT")
    summary_lines.append("="*80)
    
    timesteps = sorted(mu_dict.keys())
    
    # Compute overall average log-likelihood
    all_lls = []
    for t in timesteps:
        for ch in range(num_channels):
            ll = log_likelihood_dict[t].get(ch, np.nan)
            if not np.isnan(ll):
                all_lls.append(ll)
    avg_ll = np.mean(all_lls) if len(all_lls) else np.nan
    
    summary_lines.append("")
    summary_lines.append("--- OVERALL QUALITY ---")
    summary_lines.append(f"  Average Log-Likelihood:       {avg_ll:.4f} (higher is better)")
    summary_lines.append(f"  Number of Timesteps:          {len(timesteps)}")
    summary_lines.append(f"  Number of Channels:           {num_channels}")
    summary_lines.append(f"  Total Valid Measurements:     {len(all_lls)}")
    
    # Per-timestep summary
    summary_lines.append("")
    summary_lines.append("--- PER-TIMESTEP SUMMARY ---")
    summary_lines.append(f"{'Timestep':<10} {'Avg LogLik':<12} {'μ_quant':<30} {'μ_error':<30}")
    summary_lines.append("-" * 100)
    
    for t in timesteps:
        lls = [log_likelihood_dict[t].get(ch, np.nan) for ch in range(num_channels)]
        avg_ll_t = np.nanmean(lls)

        mu = mu_dict[t]  # (2, C)
        mu_quant_str = np.array2string(mu[0], precision=3, suppress_small=True, max_line_width=100)
        mu_error_str = np.array2string(mu[1], precision=3, suppress_small=True, max_line_width=100)
        summary_lines.append(f"{t:<10} {avg_ll_t:<12.4f} {mu_quant_str:<30} {mu_error_str:<30}")
    
    summary_lines.append("")
    summary_lines.append("="*80)
    summary_lines.append("INTERPRETATION GUIDE:")
    summary_lines.append("  • Log-Likelihood: Higher values indicate better fit to Gaussian distribution")
    summary_lines.append("  • Typical range: -10 to 10 (depends on data scale)")
    summary_lines.append("  • ρ (correlation): Correlation between quantized output and quantization error")
    summary_lines.append("  • |ρ| close to 1: Strong linear relationship")
    summary_lines.append("="*80)
    
    # Print to console
    print("\n" + "\n".join(summary_lines) + "\n")
    
    return avg_ll, "\n".join(summary_lines)


def fit_channel_wise_gaussian(
    fp16_outputs: torch.Tensor,
    quant_outputs: torch.Tensor,
    outlier_threshold: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """
    Fit channel-wise Gaussian models to joint distribution of [Quant, Error].
    
    This follows D2-DPM's approach:
    1. Compute quantization error = quant - fp16
    2. Remove outliers (4-sigma rule) based on error distribution
    3. Compute joint statistics of [quant, error] from actual data
    4. Evaluate quality using log-likelihood
    
    Args:
        fp16_outputs: FP16 model outputs, shape [N, C, H, W] or [N, 1, C, H, W]
        quant_outputs: Quantized model outputs, shape [N, C, H, W] or [N, 1, C, H, W]
        outlier_threshold: Threshold for outlier removal (default: 4 sigma)
        
    Returns:
        mu: Mean vector, shape (2, C) where mu[0] = mu_quant, mu[1] = mu_error
        cov: Covariance matrix, shape (2, 2, C) for each channel
        log_likelihoods: Dict mapping channel to log-likelihood
    """
    if fp16_outputs.ndim == 5:
        fp16_outputs = fp16_outputs.squeeze(1)
        quant_outputs = quant_outputs.squeeze(1)

    if fp16_outputs.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N, C, H, W], got shape: {tuple(fp16_outputs.shape)}")

    N, C, H, W = fp16_outputs.shape

    if fp16_outputs.dtype == torch.bfloat16:
        fp16_outputs = fp16_outputs.float()
    if quant_outputs.dtype == torch.bfloat16:
        quant_outputs = quant_outputs.float()

    # Flatten spatial + batch dims: [N, C, H, W] -> [N*H*W, C]
    fp16_flat = fp16_outputs.permute(0, 2, 3, 1).reshape(-1, C).numpy()
    quant_flat = quant_outputs.permute(0, 2, 3, 1).reshape(-1, C).numpy()
    error_flat = quant_flat - fp16_flat

    mu = np.zeros((2, C), dtype=np.float32)
    cov = np.zeros((2, 2, C), dtype=np.float32)
    log_likelihoods: Dict[int, float] = {}

    for ch in range(C):
        quant_ch = quant_flat[:, ch]
        error_ch = error_flat[:, ch]

        mean_error = float(np.mean(error_ch))
        std_error = float(np.std(error_ch))
        if std_error == 0.0:
            outliers = np.zeros_like(error_ch, dtype=bool)
        else:
            outliers = np.abs(error_ch - mean_error) > outlier_threshold * std_error

        quant_clean = quant_ch[~outliers]
        error_clean = error_ch[~outliers]

        joint = np.vstack((quant_clean, error_clean))
        mean = np.mean(joint, axis=1)
        covariance = np.cov(joint)

        mu[:, ch] = mean.astype(np.float32)
        cov[:, :, ch] = covariance.astype(np.float32)
        log_likelihoods[ch] = float(compute_log_likelihood(quant_clean, error_clean, mean, covariance))

    return mu, cov, log_likelihoods


def fit_gaussian_models(data_output_pairs_path: str, output_dir: str, outlier_threshold: float = 4.0):
    """
    Fit Gaussian models to collected FP16 and quantized outputs.
    
    This is the CORRECT implementation following D2-DPM:
    - Loads both FP16 and quantized outputs
    - Computes error = quant_output - fp16_output
    - Computes joint distribution [quant_output, error]
    - Removes outliers for robust estimation
    - Evaluates quality using log-likelihood
    
    Args:
        data_output_pairs_path: Path to data_output_pairs.pth from collect_statistics.py
        output_dir: Directory to save mu_dict.npy and cov_dict.npy
        outlier_threshold: Threshold for outlier removal (default: 4 sigma)
    """
    # Check if output files already exist (skip if found)
    mu_path = os.path.join(output_dir, "mu_dict.npy")
    cov_path = os.path.join(output_dir, "cov_dict.npy")
    
    if os.path.exists(mu_path) and os.path.exists(cov_path):
        print("="*60)
        print("⏭️  SKIPPING: Gaussian modeling")
        print("="*60)
        print(f"Output files already exist:")
        print(f"  - {mu_path}")
        print(f"  - {cov_path}")
        print("To re-run Gaussian modeling, delete these files first.")
        print("="*60)
        return
    
    print("="*60)
    print("Gaussian Modeling for Q-Drift Correction")
    print("="*60)
    
    # Load FP16 and quantized outputs
    print(f"\nLoading statistics from: {data_output_pairs_path}")
    data = torch.load(data_output_pairs_path, map_location='cpu')
    
    # Support both old and new data formats
    if 'fp16_output' in data:
        fp16_dict = data['fp16_output']
        quant_dict = data['quant_output']
    else:
        # Backward compatibility with old format (eps naming)
        fp16_dict = data.get('fp16_eps', data.get('fp16_outputs', {}))
        quant_dict = data.get('quant_eps', data.get('quant_outputs', {}))
    
    timesteps = data['timesteps']
    num_samples = data['num_samples']
    
    print(f"Loaded data:")
    print(f"  Number of samples: {num_samples}")
    print(f"  Number of timesteps: {len(timesteps)}")
    print(f"  Timesteps: {timesteps}")
    
    # Initialize dictionaries for mu, cov, and log-likelihood
    mu_dict = {}
    cov_dict = {}
    log_likelihood_dict = {}
    
    # Process each timestep
    print("\n" + "="*60)
    print("Fitting Gaussian models per timestep...")
    print(f"Outlier threshold: {outlier_threshold} sigma")
    print("="*60)
    
    for timestep in tqdm(timesteps, desc="Processing timesteps"):
        fp16_outputs = fp16_dict[timestep]
        quant_outputs = quant_dict[timestep]
        
        print(f"\nTimestep {timestep}:")
        print(f"  FP16 shape: {fp16_outputs.shape}")
        print(f"  Quant shape: {quant_outputs.shape}")
        
        # Fit channel-wise Gaussian model (one model per timestep, per-channel)
        mu, cov, ll = fit_channel_wise_gaussian(
            fp16_outputs,
            quant_outputs,
            outlier_threshold=outlier_threshold
        )
        
        mu_dict[timestep] = mu
        cov_dict[timestep] = cov
        log_likelihood_dict[timestep] = ll
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dictionaries
    mu_path = os.path.join(output_dir, "mu_dict.npy")
    cov_path = os.path.join(output_dir, "cov_dict.npy")
    
    np.save(mu_path, mu_dict)
    np.save(cov_path, cov_dict)
    
    print("\n" + "="*60)
    print("Gaussian models fitted successfully!")
    print(f"Saved mu_dict to: {mu_path}")
    print(f"Saved cov_dict to: {cov_path}")
    print("="*60)
    
    # Print quality summary
    avg_ll, summary_text = print_modeling_summary(log_likelihood_dict, mu_dict, cov_dict)
    
    # Collect detailed summary statistics
    detailed_summary_lines = []
    detailed_summary_lines.append("")
    detailed_summary_lines.append("="*60)
    detailed_summary_lines.append("DETAILED PER-TIMESTEP STATISTICS")
    detailed_summary_lines.append("="*60)
    
    for timestep in timesteps:
        mu = mu_dict[timestep]
        cov = cov_dict[timestep]

        detailed_summary_lines.append("")
        detailed_summary_lines.append(f"Timestep {timestep}:")
        detailed_summary_lines.append(f"  mu shape: {mu.shape}")
        detailed_summary_lines.append(f"  cov shape: {cov.shape}")

        C = int(mu.shape[1])
        for ch in range(C):
            sigma_quant = float(np.sqrt(max(cov[0, 0, ch], 0.0)))
            sigma_error = float(np.sqrt(max(cov[1, 1, ch], 0.0)))
            rho = float(cov[0, 1, ch] / (sigma_quant * sigma_error + 1e-10))
            ll = float(log_likelihood_dict.get(timestep, {}).get(ch, np.nan))
            detailed_summary_lines.append(
                f"  ch{ch}: μ_quant={mu[0, ch]:.6f} μ_error={mu[1, ch]:.6f} ρ={rho:.4f} ll={ll:.4f}"
            )
    
    # Print detailed summary to console
    print("\n".join(detailed_summary_lines))
    
    # Save complete analysis to text file
    analysis_path = os.path.join(output_dir, "gaussian_modeling_analysis.txt")
    with open(analysis_path, 'w') as f:
        f.write(summary_text)
        f.write("\n")
        f.write("\n".join(detailed_summary_lines))
        f.write("\n")
    
    print(f"\n✓ Complete analysis saved to: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian models for Q-Drift correction (D2-DPM implementation)"
    )
    parser.add_argument(
        "--data_output_pairs_path",
        type=str,
        required=True,
        help="Path to data_output_pairs.pth from collect_statistics.py"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for mu_dict.npy and cov_dict.npy"
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=4.0,
        help="Threshold for outlier removal in units of standard deviation (default: 4.0)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_output_pairs_path):
        raise FileNotFoundError(f"Input file not found: {args.data_output_pairs_path}")
    
    fit_gaussian_models(
        data_output_pairs_path=args.data_output_pairs_path,
        output_dir=args.output_dir,
        outlier_threshold=args.outlier_threshold,
    )


if __name__ == "__main__":
    main()
