"""
Time-Series-Library Visualization Script
=========================================
Features:
  1. Prediction vs Ground Truth comparison
  2. Training loss curves
  3. Multi-feature visualization
  4. Error analysis

Usage: python visualize.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
#                           Configuration
# ============================================================================

# Result directory (modify to your experiment result path)
RESULT_DIR = './results/long_term_forecast_ETTh1_96_96_DLinear_ETTh1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0'

# Visualization settings
SHOW_SAMPLES = 5          # Number of samples to display
FEATURE_NAMES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']  # ETTh1 features
SCALE_FACTOR = 0.7        # Scale factor for figure size (0.5 = half size, 1.0 = original)
SAVE_PLOTS = True          # Save plots to file
OUTPUT_DIR = './visualizations/'  # Output directory

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 120

# Color configuration
COLORS = {
    'true': '#2E86AB',       # Blue - Ground Truth
    'pred': '#E94F37',       # Red - Prediction
    'error': '#F6AE2D',      # Orange - Error
    'fill': '#A8DADC',       # Light blue - Fill
}


# ============================================================================
#                           Utility Functions
# ============================================================================

def load_results(result_dir):
    """Load prediction results"""
    pred_path = os.path.join(result_dir, 'pred.npy')
    true_path = os.path.join(result_dir, 'true.npy')
    metrics_path = os.path.join(result_dir, 'metrics.npy')
    
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Result file not found: {pred_path}")
    
    pred = np.load(pred_path)
    true = np.load(true_path)
    metrics = np.load(metrics_path, allow_pickle=True)
    
    print(f"Loaded results successfully!")
    print(f"   Prediction shape: {pred.shape} (samples, pred_len, features)")
    print(f"   Ground Truth shape: {true.shape}")
    print(f"   MSE: {metrics[0]:.6f}, MAE: {metrics[1]:.6f}")
    
    return pred, true, metrics


def create_output_dir():
    """Create output directory"""
    if SAVE_PLOTS and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")


# ============================================================================
#                           Visualization Functions
# ============================================================================

def plot_prediction_samples(pred, true, n_samples=5, feature_idx=0):
    """
    Display prediction results for multiple samples
    """
    n_samples = min(n_samples, len(pred))
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(14*SCALE_FACTOR, 2.2*n_samples*SCALE_FACTOR))
    if n_samples == 1:
        axes = [axes]
    
    feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f'Feature {feature_idx}'
    
    for i, ax in enumerate(axes):
        sample_idx = i * (len(pred) // n_samples)  # Uniform sampling
        
        pred_sample = pred[sample_idx, :, feature_idx]
        true_sample = true[sample_idx, :, feature_idx]
        
        time_steps = np.arange(len(pred_sample))
        
        # Plot ground truth and prediction
        ax.plot(time_steps, true_sample, color=COLORS['true'], 
                linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(time_steps, pred_sample, color=COLORS['pred'], 
                linewidth=2, label='Prediction', linestyle='--', alpha=0.8)
        
        # Fill error region
        ax.fill_between(time_steps, true_sample, pred_sample, 
                        color=COLORS['fill'], alpha=0.3, label='Error Region')
        
        # Calculate sample error
        mse = np.mean((pred_sample - true_sample) ** 2)
        mae = np.mean(np.abs(pred_sample - true_sample))
        
        ax.set_title(f'Sample #{sample_idx} | {feature_name} | MSE: {mse:.4f}, MAE: {mae:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Prediction vs Ground Truth', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_samples.png'), 
                   dpi=150, bbox_inches='tight')
        print(f"Saved: prediction_samples.png")
    
    plt.show()


def plot_all_features(pred, true, sample_idx=0):
    """
    Display prediction results for all features of a single sample
    """
    n_features = pred.shape[2]
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14*SCALE_FACTOR, 2.5*n_rows*SCALE_FACTOR))
    axes = axes.flatten()
    
    for feat_idx in range(n_features):
        ax = axes[feat_idx]
        
        pred_feat = pred[sample_idx, :, feat_idx]
        true_feat = true[sample_idx, :, feat_idx]
        time_steps = np.arange(len(pred_feat))
        
        ax.plot(time_steps, true_feat, color=COLORS['true'], 
                linewidth=2, label='Ground Truth')
        ax.plot(time_steps, pred_feat, color=COLORS['pred'], 
                linewidth=2, label='Prediction', linestyle='--')
        
        feature_name = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f'Feature {feat_idx}'
        
        mse = np.mean((pred_feat - true_feat) ** 2)
        ax.set_title(f'{feature_name} (MSE: {mse:.4f})', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Sample #{sample_idx} - All Features Prediction', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(os.path.join(OUTPUT_DIR, 'all_features.png'), 
                   dpi=150, bbox_inches='tight')
        print(f"Saved: all_features.png")
    
    plt.show()


def plot_error_analysis(pred, true):
    """
    Error analysis visualization
    """
    # Calculate average error per time step
    errors = pred - true  # (samples, pred_len, features)
    
    # Calculate various error metrics
    mse_per_step = np.mean(errors ** 2, axis=(0, 2))  # MSE per time step
    mae_per_step = np.mean(np.abs(errors), axis=(0, 2))  # MAE per time step
    mse_per_feature = np.mean(errors ** 2, axis=(0, 1))  # MSE per feature
    
    fig = plt.figure(figsize=(14*SCALE_FACTOR, 10*SCALE_FACTOR))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Error per time step
    ax1 = fig.add_subplot(gs[0, 0])
    time_steps = np.arange(len(mse_per_step))
    ax1.bar(time_steps, mse_per_step, color=COLORS['error'], alpha=0.7, label='MSE')
    ax1.plot(time_steps, mae_per_step, color=COLORS['pred'], linewidth=2, 
             marker='o', markersize=3, label='MAE')
    ax1.set_xlabel('Prediction Time Step')
    ax1.set_ylabel('Error')
    ax1.set_title('Error Analysis per Time Step', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error per feature
    ax2 = fig.add_subplot(gs[0, 1])
    feature_names = FEATURE_NAMES[:len(mse_per_feature)] if len(FEATURE_NAMES) >= len(mse_per_feature) else [f'F{i}' for i in range(len(mse_per_feature))]
    bars = ax2.bar(feature_names, mse_per_feature, color=COLORS['true'], alpha=0.7)
    ax2.set_xlabel('Feature')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE Comparison per Feature', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution histogram
    ax3 = fig.add_subplot(gs[1, 0])
    all_errors = errors.flatten()
    ax3.hist(all_errors, bins=50, color=COLORS['fill'], edgecolor=COLORS['true'], alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
    ax3.set_xlabel('Error Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution Histogram', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative error over time
    ax4 = fig.add_subplot(gs[1, 1])
    cumulative_mse = np.cumsum(mse_per_step)
    ax4.fill_between(time_steps, 0, cumulative_mse, color=COLORS['fill'], alpha=0.5)
    ax4.plot(time_steps, cumulative_mse, color=COLORS['true'], linewidth=2)
    ax4.set_xlabel('Prediction Time Step')
    ax4.set_ylabel('Cumulative MSE')
    ax4.set_title('Cumulative Error Curve', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Error Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(os.path.join(OUTPUT_DIR, 'error_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        print(f"Saved: error_analysis.png")
    
    plt.show()


def plot_prediction_overview(pred, true, n_continuous=200, feature_idx=0):
    """
    Display continuous samples prediction (concatenated into long sequence)
    """
    n_samples = min(n_continuous, len(pred))
    pred_len = pred.shape[1]
    
    # Concatenate continuous samples
    pred_concat = pred[:n_samples, :, feature_idx].flatten()
    true_concat = true[:n_samples, :, feature_idx].flatten()
    
    fig, axes = plt.subplots(2, 1, figsize=(14*SCALE_FACTOR, 8*SCALE_FACTOR), sharex=True)
    
    time_steps = np.arange(len(pred_concat))
    feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f'Feature {feature_idx}'
    
    # Top: Prediction vs Ground Truth
    ax1 = axes[0]
    ax1.plot(time_steps, true_concat, color=COLORS['true'], linewidth=1, label='Ground Truth', alpha=0.7)
    ax1.plot(time_steps, pred_concat, color=COLORS['pred'], linewidth=1, label='Prediction', alpha=0.7)
    ax1.set_ylabel('Value')
    ax1.set_title(f'Prediction Overview - {feature_name} (First {n_samples} samples)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add sample separator lines
    for i in range(0, n_samples, 10):
        ax1.axvline(x=i*pred_len, color='gray', linestyle=':', alpha=0.3)
    
    # Bottom: Error
    ax2 = axes[1]
    error = pred_concat - true_concat
    ax2.fill_between(time_steps, 0, error, 
                    where=(error >= 0), color=COLORS['pred'], alpha=0.5, label='Positive Error')
    ax2.fill_between(time_steps, 0, error, 
                    where=(error < 0), color=COLORS['true'], alpha=0.5, label='Negative Error')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Error')
    ax2.set_title('Prediction Error', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_overview.png'), 
                   dpi=150, bbox_inches='tight')
        print(f"Saved: prediction_overview.png")
    
    plt.show()


def print_summary_statistics(pred, true, metrics):
    """
    Print summary statistics
    """
    print("\n" + "="*60)
    print("Prediction Results Summary")
    print("="*60)
    print(f"Number of samples: {pred.shape[0]}")
    print(f"Prediction length: {pred.shape[1]}")
    print(f"Number of features: {pred.shape[2]}")
    print("-"*60)
    print(f"Overall MSE: {metrics[0]:.6f}")
    print(f"Overall MAE: {metrics[1]:.6f}")
    print("-"*60)
    
    # Performance per feature
    for i in range(pred.shape[2]):
        feat_mse = np.mean((pred[:, :, i] - true[:, :, i]) ** 2)
        feat_mae = np.mean(np.abs(pred[:, :, i] - true[:, :, i]))
        feature_name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'Feature {i}'
        print(f"{feature_name:8s}: MSE={feat_mse:.6f}, MAE={feat_mae:.6f}")
    
    print("="*60 + "\n")


# ============================================================================
#                           Main Function
# ============================================================================

def main():
    print("\n" + "="*60)
    print("Time-Series-Library Visualization Tool")
    print("="*60 + "\n")
    
    # Create output directory
    create_output_dir()
    
    # Load results
    pred, true, metrics = load_results(RESULT_DIR)
    
    # Print summary statistics
    print_summary_statistics(pred, true, metrics)
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    # 1. Prediction samples
    print("1. Prediction samples comparison...")
    plot_prediction_samples(pred, true, n_samples=SHOW_SAMPLES, feature_idx=6)  # OT is the 7th feature
    
    # 2. All features for single sample
    print("\n2. Single sample all features...")
    plot_all_features(pred, true, sample_idx=0)
    
    # 3. Error analysis
    print("\n3. Error analysis...")
    plot_error_analysis(pred, true)
    
    # 4. Global prediction overview
    print("\n4. Global prediction overview...")
    plot_prediction_overview(pred, true, n_continuous=100, feature_idx=6)
    
    print("\nVisualization complete!")
    if SAVE_PLOTS:
        print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
