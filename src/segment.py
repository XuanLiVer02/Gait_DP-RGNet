import numpy as np
from scipy import ndimage

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

# Split test set in this file
data = np.load('C:/Users/XuanL/PycharmProjects/PythonProject/Gait/Data/gait_r_label_normalized.npz')
final_insole = data["final_insole"]
subject_ids = data["subject_ids"]

final_grf = data["final_grf"]
final_grm = data["final_grm"]
# final_jnt = data["final_jnt"]
# print("final_insole.shape:", final_insole.shape)  # (40, 64, 16, 4673)
# print("final_grf.shape:", final_grf.shape)  # (40, 3, 4673)

unique_subjects = np.unique(subject_ids)

normalized_insole = np.zeros_like(final_insole)
normalized_grf = np.zeros_like(final_grf)
normalized_grm = np.zeros_like(final_grm)

subject_stats = {}  # Store mean/std for each subject

for subj in unique_subjects:
    idx = np.where(subject_ids == subj)[0]

    normalized_insole[..., idx], mean_insole, std_insole = normalize_data_seq(final_insole[..., idx])
    normalized_grf[..., idx], mean_grf, std_grf = normalize_data_seq(final_grf[..., idx])
    normalized_grm[..., idx], mean_grm, std_grm = normalize_data_seq(final_grm[..., idx])
    # print('mean.insole.shape', mean_insole.shape)   # mean.insole.shape (1, 64, 16, 1)
    # print('mean.grf.shape', mean_grf.shape)     # mean.grf.shape (1, 3, 1)

    subject_stats[subj] = {
        "insole": (mean_insole, std_insole),
        "grf": (mean_grf, std_grf),
        "grm": (mean_grm, std_grm),
    }

insole_train = normalized_insole
grf_train = normalized_grf
grm_train = normalized_grm

num_cycles = normalized_insole.shape[3]  # 100
time_steps = normalized_insole.shape[0]  # 2343
# print("normalized_insole.shape:", normalized_insole.shape)    # (40, 64, 16, 3925)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.filters import threshold_otsu
import matplotlib.patches as patches

# Define Angles and Constants
angle_right = -15  # Angle used in the notebook for right foot angled line
x_center = 7.5     # Midpoint of columns 0-15

normalized_insole = final_insole

# 1. Calculate the average pressure map from normalized_insole
# Average across all axes except the 64x16 dimensions
# final_insole (T, H, W, Samples)
if normalized_insole.ndim == 4:
    if normalized_insole.shape[1] == 64 and normalized_insole.shape[2] == 16:
         avg_pressure_map_r = np.mean(normalized_insole, axis=(0, -1))
    elif normalized_insole.shape[2] == 64 and normalized_insole.shape[3] == 16:
         avg_pressure_map_r = np.mean(normalized_insole, axis=(0, 1))

# Threshold only on non-zero values to avoid issues with large zero areas
non_zero_values = avg_pressure_map_r[avg_pressure_map_r > 1e-6]
if non_zero_values.size > 0:
    thresh_r = threshold_otsu(non_zero_values)
    thresh_r *= 0.6  # threshold midfoot
else:
    thresh_r = 0
agg_mask_r_approx = avg_pressure_map_r > thresh_r

# Fill holes in the mask
mask_for_labeling = ndimage.binary_fill_holes(agg_mask_r_approx)

# Retain the maximum connected domain (foot)
labeled_mask, num_features = ndimage.label(mask_for_labeling)
if num_features > 1:
    # Calculate sizes
    sizes = ndimage.sum(mask_for_labeling, labeled_mask, range(num_features + 1))
    # Find the largest area
    largest_label = sizes[1:].argmax() + 1
    mask_for_labeling = (labeled_mask == largest_label)

# Calculate boundaries based on this mask
rows_r, cols_r = np.where(mask_for_labeling)
foot_top_r = rows_r.min()
foot_bot_r = rows_r.max()
foot_left_r = cols_r.min()
foot_right_r = cols_r.max()
foot_length_r = foot_bot_r - foot_top_r + 1

# Define dividing lines based on the approximate mask
line1_intercept_r = foot_top_r + 0.54 * foot_length_r
line2_r = foot_bot_r - 0.3 * foot_length_r
slope_right = -np.tan(np.deg2rad(angle_right))

# Define median line
midline_dict_r = {}
for row in range(foot_top_r, foot_bot_r + 1):
    cols_in_row = np.where(mask_for_labeling[row, :])[0]
    if cols_in_row.size > 0:
        midline_dict_r[row] = (cols_in_row.min() + cols_in_row.max()) / 2.0
    else:
        midline_dict_r[row] = midline_dict_r.get(row - 1, x_center)

# Area IDs
region_ids_r = np.full((64, 16), -1, dtype=int)

for r in range(64):
    for c in range(16):
        if mask_for_labeling[r, c]:

            boundary_row_at_c = line1_intercept_r + slope_right * (c - x_center)
            current_mid = midline_dict_r.get(r, x_center)

            is_forefoot = r < boundary_row_at_c
            is_midfoot = boundary_row_at_c <= r < line2_r
            is_hindfoot = r >= line2_r

            is_medial = c < current_mid

            if is_forefoot:
                region_ids_r[r, c] = 0 if is_medial else 1
            elif is_midfoot:
                region_ids_r[r, c] = 2 if is_medial else 3
            elif is_hindfoot:
                region_ids_r[r, c] = 4 if is_medial else 5

foot_img_segmented = region_ids_r.copy()
# --- Visualization ---
fig, ax = plt.subplots(figsize=(6, 12))

# 1. Mask
binary_foot_mask = (foot_img_segmented != -1)
mask_cleaned = ndimage.binary_fill_holes(binary_foot_mask)
labeled_mask, num_features = ndimage.label(mask_cleaned)
if num_features > 1:
    sizes = ndimage.sum(mask_cleaned, labeled_mask, range(num_features + 1))
    largest_label = sizes[1:].argmax() + 1
    mask_cleaned = (labeled_mask == largest_label)

# 2. Foot
cmap_binary = mcolors.ListedColormap(['black', 'white'])
ax.imshow(mask_cleaned.astype(int), cmap=cmap_binary, origin='upper',
          interpolation='nearest', extent=[-0.5, 15.5, 63.5, -0.5])

# 3. Contour
ax.contour(mask_cleaned, levels=[0.5], colors='#808080', linewidths=3,
           extent=[-0.5, 15.5, 63.5, -0.5], origin='upper')

# 4. Divider
DIVIDER_COLOR = 'tomato'
DIVIDER_STYLE = '-'
DIVIDER_WIDTH = 2.5
ALPHA_VAL = 0.9

# Front-Mid
x_full = np.array([-0.5, 15.5])
y_full_1 = line1_intercept_r + slope_right * (x_full - x_center)
ax.plot(x_full, y_full_1, color=DIVIDER_COLOR, linestyle=DIVIDER_STYLE,
        linewidth=DIVIDER_WIDTH, alpha=ALPHA_VAL, label='Region Dividers')
# Front-Hind
ax.axhline(y=line2_r, color=DIVIDER_COLOR, linestyle=DIVIDER_STYLE,
           linewidth=DIVIDER_WIDTH, alpha=ALPHA_VAL)
# Medial-Lateral
valid_rows = []
valid_mids = []
for r in range(64):
    mid = midline_dict_r.get(r)
    if mid is not None:
        valid_rows.append(r)
        valid_mids.append(mid)
ax.plot(valid_mids, valid_rows, color=DIVIDER_COLOR, linestyle=DIVIDER_STYLE,
        linewidth=DIVIDER_WIDTH, alpha=ALPHA_VAL)

# 5. Region IDs
for pid in range(6):
    mask_pid = (foot_img_segmented == pid)

    if np.any(mask_pid):
        # Center of Mass
        cy, cx = ndimage.center_of_mass(mask_pid)
        txt = ax.text(cx, cy, str(pid),
                      color='tomato', fontsize=36, fontweight='bold',
                      ha='center', va='center')

# 6. Contour Box
rect = patches.Rectangle((-0.5, -0.5), 16, 64, linewidth=5, edgecolor='black', facecolor='none')
ax.add_patch(rect)

ax.set_title("Right Foot Segmentation", fontsize=16, pad=20)
ax.axis('off')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), frameon=True, fontsize=12)
plt.tight_layout()
plt.show()