import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from Gait.Week18.model_GNN_new import *
from scipy.stats import norm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def normalize_data_seq(data):
    mean_val = np.mean(data, axis=(0,-1), keepdims=True)
    std_val = np.std(data, axis=(0,-1), keepdims=True) + 1e-8
    normalized_data = (data - mean_val) / std_val
    return normalized_data, mean_val, std_val

def normalize_data_seq_with_given_min_max(data, mean_val, std_val):
    normalized_data = (data - mean_val) / (std_val + 1e-8)
    return normalized_data

def denormalize_data_seq(normalized_data, mean_val, std_val):
    return normalized_data * (std_val + 1e-8) + mean_val

def get_metrics(y_true, y_pred):
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()

    r, _ = pearsonr(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    nrmse = rmse / (np.max(y_true_flat) - np.min(y_true_flat))
    CoD = r2_score(y_true_flat, y_pred_flat, force_finite=True)
    return r, rmse, nrmse, CoD

def create_empirical_attention(X_data_raw, partition_map, num_prototypes=6):
    """
    Prior temporal attention

    X_data_raw: Raw data (N, T, H, W)
    partition_map (H, W)
    num_prototypes 6

    Returns:
        target_attention_tensor (Tensor): [T, Np]
    """
    # Raw mean activations
    print("Prior Attention")
    raw_activations = visualize_prototype_activation(
        X_data_raw, partition_map, num_prototypes
    )
    # raw_activations (40, 6)
    raw_activations = np.clip(raw_activations, 0.0, None)
    # avoid division by 0
    raw_activations += 1e-6

    target_attention_tensor = torch.tensor(raw_activations, dtype=torch.float32)

    print(f"Prior Attention, Shape: {target_attention_tensor.shape}")
    return target_attention_tensor

def visualize_prototype_activation(X_data,
                                         partition_map,
                                         num_valid_partitions):
    """
    Visualize the average prototype activation over time (0-100% Gait Cycle)
    """
    if hasattr(X_data, 'detach'): X_data = X_data.detach().cpu().numpy()
    if hasattr(partition_map, 'detach'): partition_map = partition_map.detach().cpu().numpy()

    X_data_clipped = np.clip(X_data, 0, None)
    avg_cycle = np.mean(X_data_clipped, axis=0) # (T, H, W)
    T = avg_cycle.shape[0]
    prototype_means = np.zeros((T, num_valid_partitions))

    for p_id in range(num_valid_partitions):
        mask_p = (partition_map == p_id)
        if np.sum(mask_p) > 0:
             # Mean pressure for prototype p_id
            prototype_means[:, p_id] = avg_cycle[:, mask_p].mean(axis=1)

    with plt.style.context('seaborn-v0_8-paper'):
        fig, ax = plt.subplots(figsize=(8, 5))

        # x_axis
        x_axis = np.linspace(0, 100, T)
        xlabel = "Stance Phase (%)"

        # Color
        # Fore-Med, Fore-Lat, Mid-Med, Mid-Lat, Hind-Med, Hind-Lat
        colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4', '#000000']

        labels = ['P0 (Fore-Med)', 'P1 (Fore-Lat)', 'P2 (Mid-Med)',
                    'P3 (Mid-Lat)', 'P4 (Hind-Med)', 'P5 (Hind-Lat)']

        for p_id in range(num_valid_partitions):
            ax.plot(x_axis, prototype_means[:, p_id],
                    label=labels[p_id], linewidth=2.5, alpha=0.9)

        ax.set_xlabel(xlabel, fontsize=22, fontweight='bold')
        ax.set_ylabel("Mean Pressure (a.u.)", fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, which='major', linestyle='--', alpha=0.6, color='gray', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        ax.legend(loc='upper right', frameon=False, fontsize=20)

        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

        plt.tight_layout()
        plt.show()
        return prototype_means