import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import os
from scipy import ndimage
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors

# Tools
from Gait.Week9_GNN.load_coord import coord
from Gait.Week18.segment import foot_img_segmented
from Gait.Week18.utils import *

# Models
from model.model_ds_cop import DualPath_Model, Baseline_PlainCNN, Baseline_CNN_LSTM, \
    DualPath_PathB

# 1. Data Load & process
data = np.load('C:/Users/XuanL/PycharmProjects/PythonProject/Gait/Data/gait_r_label_normalized+jnt.npz')
final_insole = data["final_insole"]
subject_ids = data["subject_ids"]
final_grf = data["final_grf"]
final_grm = data["final_grm"]
final_jnt = data["final_jnt"]
final_cop = data["final_cop"]  # (40,2,2343)

print("final_insole.shape:", final_insole.shape)  # (40, 64, 16, N_cycles)
print("final_grf.shape:", final_grf.shape)  # (40, 3, N_cycles)
unique_subjects = np.unique(subject_ids)

# Subject-wise Normalization
normalized_insole = np.zeros_like(final_insole)
normalized_cop = np.zeros_like(final_cop)
subject_stats_inputs = {}  # Restore mean/std for each subject

for subj in unique_subjects:
    idx = np.where(subject_ids == subj)[0]

    normalized_insole[..., idx], mean_insole, std_insole = normalize_data_seq(final_insole[..., idx])
    normalized_cop[..., idx], mean_cop, std_cop = normalize_data_seq(final_cop[..., idx])

    subject_stats_inputs[subj] = {
        "insole": (mean_insole, std_insole),
        "cop": (mean_cop, std_cop),
    }

insole_train = normalized_insole
cop_train = normalized_cop

num_cycles = normalized_insole.shape[3]
time_steps = normalized_insole.shape[0]
print("num_cycles:", num_cycles)

X = np.zeros((num_cycles, time_steps, 64, 16))
COP = np.zeros((num_cycles, time_steps, 2))
Y_raw = np.zeros((num_cycles, time_steps, 9))
subject_per_cycle = np.zeros(num_cycles, dtype=int)

for i in range(num_cycles):
    insole_r = insole_train[:, :, :, i]
    cop_cycle = cop_train[:, :, i]

    # final_grf shape: (Time, 3, Cycles) -> slice i -> (Time, 3)
    grf_raw = final_grf[:, :, i]
    grm_raw = final_grm[:, :, i]
    jnt_raw = final_jnt[:, :, i]

    X[i] = insole_r
    COP[i] = cop_cycle
    Y_raw[i] = np.concatenate((grf_raw, grm_raw, jnt_raw), axis=1)  # (Time, 6)
    subject_per_cycle[i] = subject_ids[i]

print("X.shape:", X.shape)
print("Y_raw.shape:", Y_raw.shape)  # (N, 40, 6)

# 2. GRF/GRM Global Channel-wise Normalization
# GRF/GRM have different magnitudes

# Subject - axis 0, Time - axis 1
# shape: (9,) -> [Fx, Fy, Fz, Mx, My, Mz, Jx, Jy, Jz]
Y_mean = np.mean(Y_raw, axis=(0, 1))
Y_std = np.std(Y_raw, axis=(0, 1))
Y_std[Y_std < 1e-6] = 1.0

print("\n--- Global Normalization Statistics ---")
feature_names = ["GRF-X", "GRF-Y", "GRF-Z", "GRM-X", "GRM-Y", "GRM-Z", "JNT-X", "JNT-Y", "JNT-Z"]
for i, name in enumerate(feature_names):
    print(f"{name}: Mean={Y_mean[i]:.2f}, Std={Y_std[i]:.2f}")

Y_pool = (Y_raw - Y_mean) / Y_std
X_pool = X
COP_pool = COP

print("X_pool shape:", X_pool.shape)
print("Y_pool shape:", Y_pool.shape)

device = torch.device('cuda')

train_loss_list = []
val_loss_list = []

def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          epochs,
          device='cuda',
          early_stopping_patience=10):
    model = model.to(device)
    best_val_main_loss = float('inf')
    wait = 0

    train_loss_history = {'total': [], 'main': []}
    val_loss_history = {'total': [], 'main': []}

    print(f"Starting training:")
    print(f"  Loss Function: Standard nn.MSELoss")
    print(f"  Scheduler: {type(scheduler).__name__}")

    for epoch in range(epochs):
        epoch_main_losses = []
        model.train()
        total_epoch_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for X_batch, CoP_batch, y_batch in train_pbar:
            X_batch, CoP_batch, y_batch = X_batch.to(device), CoP_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # Add COP
            output, _ = model(X_batch, CoP_batch)
            loss = criterion(output, y_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: NaN or Inf detected in main loss at epoch {epoch + 1}.")
                return train_loss_history, val_loss_history

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_epoch_loss += loss.item()
            epoch_main_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_train_total_loss = total_epoch_loss / len(train_loader)
        avg_train_main_loss = sum(epoch_main_losses) / len(epoch_main_losses)

        avg_val_loss= evaluate(model, val_loader, criterion)

        # 3. Scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            # 1: ReduceLROnPlateau
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            # 2: CosineAnnealingLR
            scheduler.step()

        # 4. Logs
        lrs = [g['lr'] for g in optimizer.param_groups]
        lr_log = ", ".join([f"{lr:.2e}" for lr in lrs])  # e.g., "2.00e-05, 2.00e-04"

        tqdm.write(f"Epoch {epoch + 1:03d} | Train Main: {avg_train_main_loss:.4f} | "
                   f"Val Main: {avg_val_loss:.4f} | "
                   f"LR(s): {lr_log}")

        # Early stopping
        if avg_val_loss < best_val_main_loss - 1e-5:
            best_val_main_loss = avg_val_loss
            wait = 0
            best_model_wts = model.state_dict()
            tqdm.write(f"  ✨ New best Val Main Loss: {best_val_main_loss:.5f}")
        else:
            wait += 1
            if wait >= early_stopping_patience:
                tqdm.write(
                    f"⏹ Early stopping at epoch {epoch + 1} (Val Main Loss no improvement for {early_stopping_patience} epochs)")
                break

    if 'best_model_wts' in locals():
        model.load_state_dict(best_model_wts)
        print(f"Loaded best model weights from epoch {epoch + 1 - wait} with Val Main Loss: {best_val_main_loss:.5f}")
    else:
        print("Warning: Training finished or stopped early without saving best weights.")

def evaluate(model, loader, criterion, device='cuda'):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for X_batch, CoP_batch, y_batch in loader:
            X_batch, CoP_batch, y_batch = X_batch.to(device), CoP_batch.to(device), y_batch.to(device)
            output, _ = model(X_batch, CoP_batch)
            total_loss += criterion(output, y_batch).item()
            num_batches += 1
    avg_total_loss = total_loss / num_batches

    return avg_total_loss

# 1. Hyperparameters
EPOCHS_TOTAL = 35
LR_NEW_LAYERS_ONLY = 3e-4
WEIGHT_DECAY = 5e-4
T, H, W = 40, 64, 16

# 3. Load Priors
coord_tensor = torch.from_numpy(coord).float()
coord_numpy = coord # (H, W, 2)
zero_coord_mask_numpy = (coord_numpy[..., 0] == 0) & (coord_numpy[..., 1] == 0)  # (H, W)
bg_mask_from_partition_numpy = (foot_img_segmented == -1)  # (H, W), True - BG

# 3. Incorporate masks
zero_coord_mask_full_numpy = zero_coord_mask_numpy | bg_mask_from_partition_numpy  # (H, W)

# 4. Tensor
zero_coord_mask_tensor = torch.from_numpy(zero_coord_mask_full_numpy)  # (H, W)
partition_map = torch.from_numpy(foot_img_segmented)                   # (H, W)

num_valid_partitions = len(np.unique(foot_img_segmented)) - 1
# target_attention_pattern = create_target_attention(time_steps=T, num_prototypes=num_valid_partitions)
# target_attention_tensor = torch.tensor(target_attention_pattern, dtype=torch.float32)
target_attention_tensor = create_empirical_attention(
    final_insole.transpose(3, 0, 1, 2),
    partition_map,
    num_valid_partitions
)

foot_img_flipped = np.fliplr(foot_img_segmented).copy()
def plot_attention_tensor(attention_tensor, title="Target Attention Tensor (Empirical Prior)"):
    """
    Visualize [T, Np] temporal attention
    """
    if isinstance(attention_tensor, torch.Tensor):
        data = attention_tensor.detach().cpu().numpy()
    else:
        data = attention_tensor

    # data shape: -> (Np, T)
    data_to_plot = data.T

    num_prototypes, time_steps = data_to_plot.shape
    plt.figure(figsize=(10, 4))

    # Heatmap
    im = plt.imshow(data_to_plot, cmap='viridis', aspect='auto', origin='upper',
                    interpolation='nearest')
    cbar = plt.colorbar(im)
    cbar.set_label('Mean Pressure (a.u.)')

    plt.title(title, fontsize=14)
    plt.xlabel('Stance Phase', fontsize=16)
    plt.ylabel('Anatomical Areas', fontsize=16)
    prototype_labels = ['P0 (Fore-Med)', 'P1 (Fore-Lat)', 'P2 (Mid-Med)',
                        'P3 (Mid-Lat)', 'P4 (Hind-Med)', 'P5 (Hind-Lat)']
    plt.yticks(range(num_prototypes), prototype_labels)
    plt.tight_layout()
    plt.show()
plot_attention_tensor(target_attention_tensor)

# --- 4. K-Fold ---
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=40)

feature_names = ["GRF-X", "GRF-Y", "GRF-Z", "GRM-X", "GRM-Y", "GRM-Z", "JNT_X", "JNT_Y", "JNT_Z"]
all_folds_metrics_by_feature = {name: {"r": [], "rmse": [], "nrmse": []} for name in feature_names}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_pool)):
    print(f"\n--- Fold {fold + 1}/5 ---")

    # Data Loaders
    X_tr, COP_tr, y_tr = X_pool[train_idx], COP_pool[train_idx], Y_pool[train_idx]
    X_val, COP_val, y_val = X_pool[val_idx], COP_pool[val_idx], Y_pool[val_idx]
    subj_val_fold = subject_per_cycle[val_idx]

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(COP_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(COP_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=24)

    model = DualPath_Model(
        coord_64x16=coord_tensor.to(device),
        partition_map_64x16=partition_map.to(device),
        zero_coord_mask_64x16=zero_coord_mask_tensor.to(device),
        target_attention_pattern=target_attention_tensor.to(device),
        num_prototypes=num_valid_partitions,
        cnn_dim=128,
        pos_dim=128,
        embed_dim=256,
        cop_embed_dim=128,
        proto_lstm_hidden=256,
        bottleneck_dim=128,
        global_lstm_hidden=256,
        lstm_layers=2,
        bidirectional=True,
        dropout=0.3,
        prior_confidence=0.4,
        prior_attention_weight=0.5
    ).to(device)

    # model = DualPath_PathB(
    #     coord_64x16=coord_tensor.to(device),
    #     cnn_dim=128,
    #     pos_dim=32,
    #     embed_dim=128,
    #     cop_embed_dim=32,
    #     bottleneck_dim=128,
    #     global_lstm_hidden=128,
    #     lstm_layers=2,
    #     bidirectional=True,
    #     dropout=0.35,
    # ).to(device)

    criterion = nn.MSELoss().to(device)

    params_new = []
    print(f"--- Start Training ---")
    for name, param in model.named_parameters():
        params_new.append(param)

    optimizer = torch.optim.AdamW(
        params_new, lr=LR_NEW_LAYERS_ONLY, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_TOTAL, eta_min=5e-6)
    train(model, train_loader, val_loader, criterion, optimizer,
          scheduler=scheduler,
          epochs=EPOCHS_TOTAL,
          device=device,
          early_stopping_patience=10
          )

    # Evaluation & Visualization
    model.eval()
    y_pred_list = []

    with torch.no_grad():
        for X_batch, CoP_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            CoP_batch = CoP_batch.to(device)
            y_hat_batch, _ = model(X_batch, CoP_batch)
            y_pred_list.append(y_hat_batch.cpu())

    y_pred_val_fold_norm = torch.cat(y_pred_list, dim=0).numpy()

    # Denormalize predictions & ground truth
    y_pred_val_fold_denorm = y_pred_val_fold_norm * Y_std + Y_mean
    y_val_fold_denorm = y_val * Y_std + Y_mean

    print(
        f"Fold {fold + 1} Check: Pred GRF-Z Range: {y_pred_val_fold_denorm[..., 2].min():.1f} to {y_pred_val_fold_denorm[..., 2].max():.1f}")
    print(f"Fold {fold + 1} Metrics by Feature:")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Fold {fold + 1} Exclusion Visualization (Red=Dropped, Blue=Kept)', fontsize=16)
    axes = axes.flatten()

    # Threshold to exclude outliers
    # Not using (with a higher *MIN_RANGE_THRESHOLD*)
    for feature_j in range(9):
        feature_name = feature_names[feature_j]
        true_all = y_val_fold_denorm[:, :, feature_j]
        pred_all = y_pred_val_fold_denorm[:, :, feature_j]
        r_vals, rmse_vals, nrmse_vals = [], [], []

        global_range = np.max(true_all) - np.min(true_all)
        MIN_RANGE_THRESHOLD = global_range * 0.5  # Exclude one outlier sample

        count_total = true_all.shape[0]
        count_dropped = 0
        vis_dropped_indices = []
        vis_kept_indices = []

        # Record worst case
        current_feature_nrmses = []

        for sample_i in range(count_total):
            true_seq = true_all[sample_i]
            pred_seq = pred_all[sample_i]
            local_range = np.max(true_seq) - np.min(true_seq)

            if local_range < MIN_RANGE_THRESHOLD:
                count_dropped += 1
                if len(vis_dropped_indices) < 3:
                    vis_dropped_indices.append(sample_i)
                # placeholder for worst case
                current_feature_nrmses.append(0.0)
                continue
            else:
                if len(vis_kept_indices) < 1:
                    vis_kept_indices.append(sample_i)

            r, rmse, nrmse_ratio, CoD = get_metrics(true_seq, pred_seq)
            nrmse_percent = nrmse_ratio * 100

            r_vals.append(r)
            rmse_vals.append(rmse)
            nrmse_vals.append(nrmse_percent)
            current_feature_nrmses.append(nrmse_percent)

        avg_r = np.nanmean(r_vals) if r_vals else 0.0
        avg_rmse = np.nanmean(rmse_vals) if rmse_vals else 0.0
        avg_nrmse = np.nanmean(nrmse_vals) if nrmse_vals else 0.0
        drop_rate = (count_dropped / count_total) * 100

        all_folds_metrics_by_feature[feature_name]["r"].append(avg_r)
        all_folds_metrics_by_feature[feature_name]["rmse"].append(avg_rmse)
        all_folds_metrics_by_feature[feature_name]["nrmse"].append(avg_nrmse)

        print(f"  {feature_name}: R={avg_r:.4f}, RMSE={avg_rmse:.4f}, NRMSE={avg_nrmse:.2f}%")
        print(f"     -> Thresh: {MIN_RANGE_THRESHOLD:.2f}, Dropped: {count_dropped}/{count_total} ({drop_rate:.1f}%)")

    if feature_j < 6:  # GRF & GRM
        ax = axes[feature_j]
        ax.set_title(f"{feature_name} (Drop: {drop_rate:.1f}%)")
        if vis_kept_indices:
            ax.plot(true_all[vis_kept_indices[0]], color='blue', alpha=0.6, label='Kept', linewidth=2)
        for idx in vis_dropped_indices:
            ax.plot(true_all[idx], color='red', alpha=0.4, linewidth=1, label='Dropped')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize='small')

    else:
        if len(current_feature_nrmses) > 0:
            worst_sample_idx = np.argmax(current_feature_nrmses)
            worst_nrmse_val = current_feature_nrmses[worst_sample_idx]
            if worst_nrmse_val > 20.0:
                print(
                    f"     [Warning] Worst Sample for {feature_name}: Idx {worst_sample_idx}, NRMSE={worst_nrmse_val:.2f}%")

    plt.tight_layout()
    plt.show()

# Summary & Visualization
print("\n" + "=" * 50)
print("--- 5-Fold Cross Validation Summary (Mean ± Std) ---")
print("=" * 50)

for feature_name in feature_names:
    r_list = all_folds_metrics_by_feature[feature_name]["r"]
    rmse_list = all_folds_metrics_by_feature[feature_name]["rmse"]
    nrmse_list = all_folds_metrics_by_feature[feature_name]["nrmse"]

    # Mean
    avg_r = np.nanmean(r_list)
    avg_rmse = np.nanmean(rmse_list)
    avg_nrmse = np.nanmean(nrmse_list)

    # SD
    std_r = np.nanstd(r_list)
    std_rmse = np.nanstd(rmse_list)
    std_nrmse = np.nanstd(nrmse_list)

    print(f"Feature: {feature_name}")
    print(f"  R     : {avg_r:.4f} ± {std_r:.4f}")
    print(f"  RMSE  : {avg_rmse:.4f} ± {std_rmse:.4f}")
    print(f"  NRMSE : {avg_nrmse:.2f}% ± {std_nrmse:.2f}%")
    print("-" * 30)

print("\n--- Overall Performance (All Features Averaged) ---")
all_r = [val for name in feature_names for val in all_folds_metrics_by_feature[name]["r"]]
all_nrmse = [val for name in feature_names for val in all_folds_metrics_by_feature[name]["nrmse"]]

print(f"Global Mean R     : {np.nanmean(all_r):.4f}")
print(f"Global Mean NRMSE : {np.nanmean(all_nrmse):.2f}%")

# Visualize last fold
if y_pred_val_fold_norm.shape[0] > 0:
    x_axis_plot = np.arange(y_val_fold_denorm.shape[1])
    plt.figure(figsize=(18, 12))
    plt.suptitle('Fold 5 Validation: Predicted & Real (Mean ± 1 StdDev)', fontsize=16)

    plot_details = [
        (y_val_fold_denorm[:, :, 0], y_pred_val_fold_denorm[:, :, 0], feature_names[0]),  # GRF-X
        (y_val_fold_denorm[:, :, 3], y_pred_val_fold_denorm[:, :, 3], feature_names[3]),  # GRM-X
        (y_val_fold_denorm[:, :, 1], y_pred_val_fold_denorm[:, :, 1], feature_names[1]),  # GRF-Y
        (y_val_fold_denorm[:, :, 4], y_pred_val_fold_denorm[:, :, 4], feature_names[4]),  # GRM-Y
        (y_val_fold_denorm[:, :, 2], y_pred_val_fold_denorm[:, :, 2], feature_names[2]),  # GRF-Z
        (y_val_fold_denorm[:, :, 5], y_pred_val_fold_denorm[:, :, 5], feature_names[5]),  # GRM-Z
    ]

    # Plot 1,4 (X) | Plot 2,5 (Y) | Plot 3,6 (Z)
    subplot_indices = [1, 4, 2, 5, 3, 6]

    for i, (real_all, pred_all, title) in enumerate(plot_details):
        plt.subplot(3, 2, subplot_indices[i])

        real_mean, real_std = np.mean(real_all, axis=0), np.std(real_all, axis=0)
        pred_mean, pred_std = np.mean(pred_all, axis=0), np.std(pred_all, axis=0)

        plt.plot(x_axis_plot, real_mean, color='blue', label='Real Mean')
        plt.fill_between(x_axis_plot, real_mean - real_std, real_mean + real_std, color='blue', alpha=0.2)
        plt.plot(x_axis_plot, pred_mean, color='red', label='Pred Mean')
        plt.fill_between(x_axis_plot, pred_mean - pred_std, pred_mean + pred_std, color='red', alpha=0.2)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()