import numpy as np
from scipy.io import loadmat
from scipy import interpolate
from scipy.signal import butter, filtfilt
import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
file_dir = base_dir/"data"/"raw"
processed_dir = base_dir/"data"/"processed"
processed_data_dir = processed_dir/"gait_r_label_normalized+jnt.npz"

def lowpass_filter(data, cutoff=10, fs=100, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Butterworth Filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data
def process_gait_cycles(insole_data, force_data, strikes, offs, strikes_force, offs_force, pcami=False):
    """
    Extract and normalize stance phases for insole and force plate

    Parameters:
    insole_data -- array of insole measurements (time x channels)
    force_data -- array of force plate measurements (time x channels)
    strikes -- heel strike indices
    offs -- toe off indices

    Returns:
    norm_insole -- normalized insole stance phase (100 points per cycle)
    avg_insole -- average of normalized insole data
    std_insole -- standard deviation of normalized insole data
    norm_force -- normalized force plate stance phase (100 points per cycle)
    avg_force -- average of normalized force plate data
    std_force -- standard deviation of normalized force plate data
    """
    num_cycles = min(len(strikes), len(offs), len(strikes_force), len(offs_force))
    insole_stance = []
    force_stance = []
    # Extract stance phases for each cycle
    for i in range(num_cycles):
        if strikes[i] < offs[i]:
            insole_stance.append(insole_data[strikes[i]:offs[i], :])
    for i in range(num_cycles):
        if strikes_force[i] < offs_force[i]:
            force_stance.append(force_data[strikes_force[i]:offs_force[i], :])

#    insole_for_pca = np.vstack(insole_stance)
#    force_for_pca = np.vstack(force_stance)
#    print('123', insole_for_pca.shape, force_for_pca.shape)
#    if pcami:
#        n_components = 5
#        top_k_features = 10
#        X_reduced = pcami_cycle(insole_for_pca, force_for_pca, n_components=n_components, top_k_features=top_k_features)
#        split_indices = np.cumsum(num_cycles)
#        X_splits = np.split(X_reduced, split_indices[:-1])
#        insole_stance = X_splits

    num_valid = len(insole_stance)

    if num_valid == 0:
        raise ValueError('No valid stance phases found')

    # Initialization
    num_points = 40
    raw_insole = np.zeros((num_points, insole_data.shape[1], num_valid))
    raw_force = np.zeros((num_points, force_data.shape[1], num_valid))

    for i in range(num_valid):  # Every step
        insole_cycle = insole_stance[i]     # 30, num_features
        force_cycle = force_stance[i]   # 75,3

        t_insole = np.arange(insole_cycle.shape[0]) * 0.025
        t_force_full = np.arange(force_cycle.shape[0]) * 0.01

        valid_force_idx = t_force_full <= t_insole[-1]
        t_force = t_force_full[valid_force_idx]
        force_cycle = force_cycle[valid_force_idx, :]

        standard_time_insole = np.linspace(0, t_insole[-1], num_points)
        standard_time_force = np.linspace(0, t_force[-1], num_points)
        for j in range(insole_cycle.shape[1]):
            f_insole = interpolate.interp1d(t_insole, insole_cycle[:, j], kind='cubic', fill_value='extrapolate')
            raw_insole[:, j, i] = f_insole(standard_time_insole)
            raw_insole[:, j, i] = np.clip(f_insole(standard_time_insole), 0, 4080)
        for j in range(force_cycle.shape[1]):
            f_force = interpolate.interp1d(t_force, force_cycle[:, j], kind='cubic', fill_value='extrapolate')
            raw_force[:, j, i] = f_force(standard_time_force)

    avg_insole = np.mean(raw_insole, axis=2)
    std_insole = np.std(raw_insole, axis=2)

    avg_force = np.mean(raw_force, axis=2)
    std_force = np.std(raw_force, axis=2)

    return raw_insole, avg_insole, std_insole, raw_force, avg_force, std_force
def process_file(filepath):
    data = loadmat(filepath, struct_as_record=False, squeeze_me=True)

    # Get gait data
    gait_insole = data['gait_insole']
    gait_trackers = data['gait_trackers']

    idx = 99
    if np.any(np.isnan(gait_trackers.force_plate_ds_r[idx, :])):
        gait_trackers.force_plate_ds_r[idx, :] = (gait_trackers.force_plate_ds_r[idx - 1, :] + gait_trackers.force_plate_ds_r[idx + 1, :]) / 2.0
    gait_trackers.force_plate_ds_r = np.nan_to_num(gait_trackers.force_plate_ds_r)

    grf_r = lowpass_filter(gait_trackers.force_plate_ds_r[:, 0:3], cutoff=10, fs=100, order=2)
    grm_r = lowpass_filter(gait_trackers.force_plate_ds_r[:, 3:6], cutoff=10, fs=100, order=2)
    grf_l = lowpass_filter(gait_trackers.force_plate_ds_l[:, 0:3], cutoff=10, fs=100, order=2)
    grm_l = lowpass_filter(gait_trackers.force_plate_ds_l[:, 3:6], cutoff=10, fs=100, order=2)

    # Right foot GRF
    _, _, _, norm_right_grf, avg_right_grf, std_right_grf = process_gait_cycles(
        gait_insole.pp_r.reshape(-1, 1),
        grf_r,
        gait_insole.strike_r,
        gait_insole.off_r,
        gait_trackers.strike_r,
        gait_trackers.off_r)
    # Left foot GRF
    _, _, _, norm_left_grf, avg_left_grf, std_left_grf = process_gait_cycles(
        gait_insole.pp_l.reshape(-1, 1),
        grf_l,
        gait_insole.strike_l,
        gait_insole.off_l,
        gait_trackers.strike_l,
        gait_trackers.off_l)

    # Process Force Plate GRM
    # Right foot GRM
    _, _, _, norm_right_grm, avg_right_grm, std_right_grm = process_gait_cycles(
        gait_insole.pp_r.reshape(-1, 1),
        grm_r,
        gait_insole.strike_r,
        gait_insole.off_r,
        gait_trackers.strike_r,
        gait_trackers.off_r
    )

    # Left foot GRM
    _, _, _, norm_left_grm, avg_left_grm, std_left_grm = process_gait_cycles(
        gait_insole.pp_l.reshape(-1, 1),
        grm_l,
        gait_insole.strike_l,
        gait_insole.off_l,
        gait_trackers.strike_l,
        gait_trackers.off_l
    )

    # Right joint angles
    _, _, _, norm_right_jnt, avg_right_jnt, std_right_jnt = process_gait_cycles(
        gait_insole.pp_r.reshape(-1, 1),
        gait_trackers.jnt_angles_r,
        gait_insole.strike_r,
        gait_insole.off_r,
        gait_trackers.strike_r,
        gait_trackers.off_r
    )

    # Used in CNN
    norm_right_insole_all, avg_right_insole_all, std_right_insole_all, _, _, _ = process_gait_cycles(
        gait_insole.insole_r,
        grf_r,
        gait_insole.strike_r,
        gait_insole.off_r,
        gait_trackers.strike_r,
        gait_trackers.off_r)

    norm_left_insole_all, avg_left_insole_all, std_v_insole_all, _, _, _ = process_gait_cycles(
        gait_insole.insole_l,
        grf_l,
        gait_insole.strike_l,
        gait_insole.off_l,
        gait_trackers.strike_l,
        gait_trackers.off_l)

    # Right foot COP
    _, _, _, norm_right_cop_fp, avg_right_cop_fp, std_right_cop_fp = process_gait_cycles(
        gait_insole.pp_r.reshape(-1, 1),
        gait_trackers.force_plate_ds_r[:, 6:8],
        gait_insole.strike_r,
        gait_insole.off_r,
        gait_trackers.strike_r,
        gait_trackers.off_r
    )

    # Left foot COP
    _, _, _, norm_left_cop_fp, avg_left_cop_fp, std_left_cop_fp = process_gait_cycles(
        gait_insole.pp_l.reshape(-1, 1),
        gait_trackers.force_plate_ds_l[:, 6:8],
        gait_insole.strike_l,
        gait_insole.off_l,
        gait_trackers.strike_l,
        gait_trackers.off_l
    )

    # Right foot insole COP
    right_cop_data = np.column_stack((gait_insole.cop_x_r, gait_insole.cop_y_r))
    norm_right_cop_insole, avg_right_cop_insole, std_right_cop_insole, _, _, _ = process_gait_cycles(
        right_cop_data,
        gait_trackers.force_plate_ds_r[:, 6:8],  # force plate COP
        gait_insole.strike_r,
        gait_insole.off_r,
        gait_trackers.strike_r,
        gait_trackers.off_r
    )

    # Left foot insole COP
    left_cop_data = np.column_stack((gait_insole.cop_x_l, gait_insole.cop_y_l))
    norm_left_cop_insole, avg_left_cop_insole, std_left_cop_insole, _, _, _ = process_gait_cycles(
        left_cop_data,
        gait_trackers.force_plate_ds_l[:, 6:8],  # force plate COP
        gait_insole.strike_l,
        gait_insole.off_l,
        gait_trackers.strike_l,
        gait_trackers.off_l
    )

    norm_right_insole_all = norm_right_insole_all.reshape((40,64,16,min(len(gait_insole.strike_r), len(gait_insole.off_r), len(gait_trackers.strike_r), len(gait_trackers.off_r))), order = 'F')     # TIme points, 64, 16, Steps
    norm_right_insole_all_copy = norm_right_insole_all.copy()
    norm_right_insole_all[:, :32, :, :] = norm_right_insole_all_copy[:, :32, :, :][:, ::-1, :, :]
    norm_right_insole_all = norm_right_insole_all[:, :, ::-1, :]
    print("norm_right_insole_all shape:", norm_right_insole_all.shape)

    # Left foot
    norm_left_insole_all = norm_left_insole_all.reshape((40,64,16,min(len(gait_insole.strike_l), len(gait_insole.off_l), len(gait_trackers.strike_l), len(gait_trackers.off_l))), order = 'F')     # TIme points, 64, 16, Steps
    norm_left_insole_all_copy = norm_left_insole_all.copy()
    norm_left_insole_all[:, :32, :, :] = norm_left_insole_all_copy[:, :32, :, :][:, ::-1, :, :]
    norm_left_insole_all = norm_left_insole_all[:, :, ::-1, :]

# RightHalf = Uniform/Normal Distribution
#     left_half = norm_right_insole_all[:, :, :8, :]  # (100,64,8,steps)
#     vmin, vmax = left_half.min(), left_half.max()
#     right_half = np.random.uniform(low=vmin, high=vmax, size=norm_right_insole_all[:, :, 8:, :].shape)
#     mean, std = left_half.mean(), left_half.std()
#     right_half = np.random.normal(loc=mean, scale=std, size=norm_right_insole_all[:, :, 8:, :].shape)
#     right_half = np.clip(right_half, a_min=0, a_max=None)
#     norm_right_insole_all = np.concatenate([left_half, right_half], axis=2)

# UpperHalf=0
#     bottom_half = norm_right_insole_all[:, 62:, :, :]  # (100,32,16,steps)
#     upper_half = np.zeros_like(norm_right_insole_all[:, :62, :, :])  # (100,32,16,steps)
#     upper_half[:] = -2000
#     norm_right_insole_all = np.concatenate([upper_half, bottom_half], axis=1)
# RightHalf=0
#     left_half = norm_right_insole_all[:, :, :4, :]  # (100,64,8,steps)
#     right_half = np.zeros_like(norm_right_insole_all[:, :, 4:, :])  # (100,64,8,steps)
#     norm_right_insole_all = np.concatenate([left_half, right_half], axis=2)
#LeftHalf=0
    # right_half = norm_right_insole_all[:, :, 8:, :]  # (100,64,8,steps)
    # left_half = np.zeros_like(norm_right_insole_all[:, :, :8, :])  # (100,64,8,steps)
    # norm_right_insole_all = np.concatenate([left_half, right_half], axis=2)
#K interval
    # k=2
    # norm_right_insole_all = norm_right_insole_all[:, ::k, ::k, :]
#Center=0
    # center = norm_right_insole_all[:, :, 4:12, :]  # (100,64,8,steps)
    # center_zero = np.zeros_like(center)  # (100,64,8,steps)
    # norm_right_insole_all = np.concatenate([norm_right_insole_all[:, :, :4, :], center_zero, norm_right_insole_all[:, :, 12:, :]], axis=2)
# Average to 32*8 & 2 interval
#     x = norm_right_insole_all
#     x = x.reshape(100, 32, 2, 8, 2, -1)  # (time, 32,2, 8,2, steps)
#     print("x shape before reshape:", x.shape)  # (100, 64, 16, num_steps)
#     x = x.mean(axis=(2, 4))
#     norm_right_insole_all = x  # (100, 32, 8, num_steps)
#     norm_right_insole_all = norm_right_insole_all[:, ::2, ::2, :]
# Average to 16*4, then randomly choose 8*2 from 16*4
#     x = norm_right_insole_all
#     x = x.reshape(100, 16, 4, 4, 4, -1)
#     x = x.mean(axis=(2, 4))
#     print("x shape before random choose:", x.shape)  # (100, 16, 4, num_steps)
#     norm_right_insole_all = x  # (100, 16, 4, num_steps)
#     indices_w = np.sort(np.random.choice(16, 8, replace=False))
#     indices_h = np.sort(np.random.choice(4, 2, replace=False))
#     norm_right_insole_all = norm_right_insole_all[:, indices_w, :][:, :, indices_h, :]

    print("norm_right_insole_all shape:", norm_right_insole_all.shape)  # (100, 64, 16, num_steps)

    # Visualize one example
    # plt.figure(figsize=(10, 10))
    # plt.imshow(norm_right_insole_all[50, :, :, 0],
    #            cmap=mpl.colors.ListedColormap(mpl.colormaps['jet'](np.linspace(0, 1, 256))))
    # plt.xlabel('Width (16 pixels)')
    # plt.ylabel('Height (64 pixels)')
    # plt.colorbar(label='Intensity')
    # plt.title('Footstep Resized Image')
    # plt.show()

    return norm_right_insole_all, norm_right_grf, norm_right_grm, norm_right_cop_insole, norm_right_jnt, norm_left_insole_all, norm_left_grf, norm_left_grm

file_dir = "C:/Users/XuanL/Documents/MATLAB/Gait Analysis/NewData0805"
file_list = [
             "ABLE1_112024_1.0.mat", "ABLE2_030525_1.0.mat", "ABLE3_030525_1.0.mat", "ABLE4_030525_1.0.mat",
             "ABLE1_112024_0.75.mat", "ABLE2_030525_0.75.mat", "ABLE3_030525_0.75.mat", "ABLE4_030525_0.75.mat",
             "ABLE1_112024_1.5.mat", "ABLE2_030525_1.5.mat", "ABLE3_030525_1.5.mat",
             "ABLE2_030525_2.0.mat", "ABLE3_030525_2.0.mat"
             ]
# file_list = ["ABLE1_112024_1.0.mat"]
# file_list = ["ABLE1_112024_1.0.mat", "ABLE2_030525_1.0.mat", "ABLE3_030525_1.0.mat", "ABLE4_030525_1.0.mat"]

all_insole = []
all_grf = []
all_grm = []
all_cop = []
all_jnt = []
all_insole_left = []
all_grf_left = []
all_grm_left = []

# Right Foot
for fname in file_list:
    if fname in ("ABLE1_112024_1.0.mat", "ABLE1_112024_0.75.mat", "ABLE1_112024_1.5.mat"):
        height=1.7526
        weight=77.11
    if fname in ("ABLE2_030525_1.0.mat", "ABLE2_030525_0.75.mat", "ABLE2_030525_1.5.mat", "ABLE2_030525_2.0.mat"):
        height=1.8288
        weight=78.47
    if fname in ("ABLE3_030525_1.0.mat", "ABLE3_030525_0.75.mat", "ABLE3_030525_1.5.mat", "ABLE3_030525_2.0.mat"):
        height=1.778
        weight=71.7
    if fname in ("ABLE4_030525_1.0.mat", "ABLE4_030525_0.75.mat"):
        height=1.6002
        weight=70.3

subject_ids = []
# Right Foot
for subj_id, fname in enumerate(file_list):
    path = os.path.join(file_dir, fname)
    norm_right_insole_all, norm_right_grf, norm_right_grm, norm_right_cop_insole, norm_right_jnt, _, _, _ = process_file(path)
    norm_right_grf = norm_right_grf / weight * 100  # Normalize to body weight
    norm_right_grm = norm_right_grm / (weight * height) * 100  # Normalize to body weight * height
    all_insole.append(norm_right_insole_all)
    all_grf.append(norm_right_grf)
    all_grm.append(norm_right_grm)
    all_cop.append(norm_right_cop_insole)
    all_jnt.append(norm_right_jnt)

    # How many steps for the subject?
    n_steps = norm_right_insole_all.shape[-1]
    subject_ids.extend([subj_id] * n_steps)
    print('subject_ids length:', len(subject_ids))

subject_ids = np.array(subject_ids)

final_insole = np.concatenate(all_insole, axis=3)
final_grf = np.concatenate(all_grf, axis=2)
final_grm = np.concatenate(all_grm, axis=2)
final_cop = np.concatenate(all_cop, axis=2)
final_jnt = np.concatenate(all_jnt, axis=2)
# final_insole_left = np.concatenate(all_insole_left, axis=3)
# final_grf_left = np.concatenate(all_grf_left, axis=2)
# final_grm_left = np.concatenate(all_grm_left, axis=2)

print("insole shape:", final_insole.shape)
print("grf shape:", final_grf.shape)
print("grm shape:", final_grm.shape)
print("cop shape:", final_cop.shape)
print("jnt shape:", final_jnt.shape)

np.savez(processed_data_dir,
         final_insole=final_insole,
         final_grf=final_grf,
         final_grm=final_grm,
         final_cop=final_cop,
         final_jnt=final_jnt,
         subject_ids=subject_ids
)