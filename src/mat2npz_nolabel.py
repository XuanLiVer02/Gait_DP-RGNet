import numpy as np
from scipy.io import loadmat
from scipy import interpolate
import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
file_dir = base_dir/"data"/"raw"
processed_data_dir = base_dir/"data"/"processed"
file_dir = processed_data_dir/"gait_40_r_withcop_SSL.npz"

def process_gait_cycles(insole_data, strikes, offs):
    """
    Extract and normalize stance phases for insole and force plate

    Parameters:
    insole_data -- array of insole measurements (time x channels)
    force_data -- array of force plate measurements (time x channels)
    strikes -- heel strike indices
    offs -- toe off indices

    """
    num_cycles = min(len(strikes), len(offs))
    insole_stance = []
    # Extract stance phases for each cycle
    for i in range(num_cycles):
        if strikes[i] < offs[i]:
            insole_stance.append(insole_data[strikes[i]:offs[i], :])
    num_valid = len(insole_stance)

    num_points = 40
    raw_insole = np.zeros((num_points, insole_data.shape[1], num_valid))
    for i in range(num_valid):  # Every step
        insole_cycle = insole_stance[i]
        t_insole = np.arange(insole_cycle.shape[0]) * 0.025
        standard_time_insole = np.linspace(0, t_insole[-1], num_points)
        for j in range(insole_cycle.shape[1]):
            if len(t_insole) < 4:
                # linear interp if time points < 4
                f_insole = interpolate.interp1d(t_insole, insole_cycle[:, j], kind='linear', fill_value='extrapolate')
            else:
                f_insole = interpolate.interp1d(t_insole, insole_cycle[:, j], kind='cubic', fill_value='extrapolate')
                raw_insole[:, j, i] = f_insole(standard_time_insole)
                raw_insole[:, j, i] = np.clip(f_insole(standard_time_insole), 0, 4080)
    return raw_insole

def process_balance_data(insole_data):
    num_points = 40
    total_frames = insole_data.shape[0]
    num_cycles = total_frames // num_points
    t_insole = np.arange(insole_data.shape[0]) * 0.025
    standard_time_insole = np.linspace(0, t_insole[-1], num_points)
    raw_insole = np.zeros((num_points, insole_data.shape[1], num_cycles))
    for j in range(insole_data.shape[1]):
        kind = 'linear' if len(t_insole) < 4 else 'cubic'
        f_insole = interpolate.interp1d(t_insole, insole_data[:, j], kind=kind, fill_value='extrapolate')
        raw_insole[:, j, 0] = f_insole(standard_time_insole)
        raw_insole[:, j, 0] = np.clip(f_insole(standard_time_insole), 0, 4080)
    return raw_insole

def process_file(filepath):
    data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    # Get gait data
    gait_insole = data['gait_insole']

    # Used in CNN
    norm_right_insole_all = process_gait_cycles(
        gait_insole.insole_r,
        gait_insole.strike_r,
        gait_insole.off_r
    )
    # Right foot insole COP
    right_cop_data = np.column_stack((gait_insole.cop_x_r, gait_insole.cop_y_r))
    norm_right_cop_insole = process_gait_cycles(
        right_cop_data,
        gait_insole.strike_r,
        gait_insole.off_r,
    )
    # norm_left_insole_all = process_gait_cycles(
    #     gait_insole.insole_l,
    #     gait_insole.strike_l,
    #     gait_insole.off_l
    # )

    num_windows_r = norm_right_insole_all.shape[2]    # For subject 1, 1m/s, (40/100, 1024, 199)
    # num_windows_l = norm_left_insole_all.shape[2]
    # print("Number of windows:", num_windows_r)

    norm_right_insole_all = norm_right_insole_all.reshape((40,64,16,num_windows_r), order = 'F')     # TIme points, 64, 16, Steps
    norm_right_insole_all_copy = norm_right_insole_all.copy()
    norm_right_insole_all[:, :32, :, :] = norm_right_insole_all_copy[:, :32, :, :][:, ::-1, :, :]
    norm_right_insole_all = norm_right_insole_all[:, :, ::-1, :]

    # Left foot
    # norm_left_insole_all = norm_left_insole_all.reshape((40,64,16,num_windows_l), order = 'F')     # TIme points, 64, 16, Steps
    # norm_left_insole_all_copy = norm_left_insole_all.copy()
    # norm_left_insole_all[:, :32, :, :] = norm_left_insole_all_copy[:, :32, :, :][:, ::-1, :, :]
    # norm_left_insole_all = norm_left_insole_all[:, :, ::-1, :]

    print("norm_right_insole_all shape:", norm_right_insole_all.shape)  # (40, 64, 16, num_steps)
    print("norm_right_cop_insole shape:", norm_right_cop_insole.shape)
    # print("norm_left_insole_all shape:", norm_left_insole_all.shape)

    # return norm_right_insole_all, norm_left_insole_all
    return norm_right_insole_all, norm_right_cop_insole

def process_balance_file(filepath):
    data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    # Get gait data
    gait_insole = data['gait_insole']

    # Used in CNN
    norm_right_insole_all = process_balance_data(
        gait_insole.insole_r
    )

    num_windows_r = norm_right_insole_all.shape[2]    # For subject 1, 1m/s, (40/100, 1024, 199)

    norm_right_insole_all = norm_right_insole_all.reshape((40,64,16,num_windows_r), order = 'F')     # TIme points, 64, 16, Steps
    norm_right_insole_all_copy = norm_right_insole_all.copy()
    norm_right_insole_all[:, :32, :, :] = norm_right_insole_all_copy[:, :32, :, :][:, ::-1, :, :]
    norm_right_insole_all = norm_right_insole_all[:, :, ::-1, :]

    # Left foot
    # norm_left_insole_all = norm_left_insole_all.reshape((40,64,16,num_windows_l), order = 'F')     # TIme points, 64, 16, Steps
    # norm_left_insole_all_copy = norm_left_insole_all.copy()
    # norm_left_insole_all[:, :32, :, :] = norm_left_insole_all_copy[:, :32, :, :][:, ::-1, :, :]
    # norm_left_insole_all = norm_left_insole_all[:, :, ::-1, :]

    print("norm_right_insole_all shape:", norm_right_insole_all.shape)  # (40, 64, 16, num_steps)
    # print("norm_left_insole_all shape:", norm_left_insole_all.shape)

    # Visualize one example
    # plt.figure(figsize=(10, 10))
    # plt.imshow(norm_right_insole_all[20, :, :, 0],
    #            cmap=mpl.colors.ListedColormap(mpl.colormaps['jet'](np.linspace(0, 1, 256))))
    # plt.xlabel('Width (16 pixels)')
    # plt.ylabel('Height (64 pixels)')
    # plt.colorbar(label='Intensity')
    # plt.title('Footstep Resized Image')
    # plt.show()

    # return norm_right_insole_all, norm_left_insole_all
    return norm_right_insole_all

# file_dir = "C:/Users/XuanL/Documents/MATLAB/Gait Analysis/NewData0805"

file_list = [
             # "ABLE1_112024_1.0.mat", "ABLE2_030525_1.0.mat", "ABLE3_030525_1.0.mat", "ABLE4_030525_1.0.mat",
             # "ABLE1_112024_0.75.mat", "ABLE2_030525_0.75.mat", "ABLE3_030525_0.75.mat", "ABLE4_030525_0.75.mat",
             # "ABLE1_112024_1.5.mat", "ABLE2_030525_1.5.mat", "ABLE3_030525_1.5.mat",
             # "ABLE2_030525_2.0.mat", "ABLE3_030525_2.0.mat"
             'ABLE1_112024_0.5_1.0.mat', 'ABLE1_112024_1.0_0.5.mat',
             # 'ABLE5_010325_ols_ab_r.mat', 'ABLE5_010325_ols_ab_r_2.mat', 'ABLE5_010325_ols_ab_r_3.mat', 'ABLE5_010325_ols_h_l.mat', 'ABLE5_010325_ols_h_r.mat',
             # 'ABLE5_010325_ols_l.mat', 'ABLE5_010325_ols_l_2.mat', 'ABLE5_010325_ols_r.mat', 'ABLE5_010325_ols_r_2.mat', 'ABLE5_010325_roll_lr.mat',
             'ABLE5_071724_0.6.mat', 'ABLE5_080624_0.6.mat', 'ABLE5_080624_0.6_1.0.mat', 'ABLE5_080624_0.6_1.4.mat', 'ABLE5_080624_0.6_1.8.mat', 'ABLE5_080624_1.0_0.6.mat', 'ABLE5_080624_1.4_0.6.mat',
                 'ABLE5_080624_1.8_0.6.mat', 'ABLE5_093024_0.6.mat', 'ABLE5_102224_0.6.mat', 'ABLE5_102324_0.6.mat', 'ABLE5_102324_1.2.mat', 'ABLE5_102324_1.8.mat', 'ABLE5_102324_1.8_2.mat',
                 'ABLE5_102324_2.4.mat', 'ABLE5_102324_2.4_2.mat', 'ABLE5_102324_ab_l_0.6.mat', 'ABLE5_102324_ab_l_1.2.mat', 'ABLE5_102324_ab_l_1.8.mat', 'ABLE5_102324_ab_l_2.4.mat',
                 'ABLE5_102324_ab_l_2.4_2.mat', 'ABLE6_071724_0.6.mat', 'ABLE6_071824_0.6.mat', 'ABLE6_071824_0.6_2.mat', 'ABLE6_071824_0.6_3.mat', 'ABLE6_073024_0.6.mat',
                 'ABLE6_073024_0.6_0.6-1.8.mat', 'ABLE6_073124_0.6.mat', 'ABLE6_073124_0.6_0.6-1.6.mat', 'ABLE6_073124_0.6_1.0.mat', 'ABLE6_073124_0.6_1.4.mat', 'ABLE6_073124_0.6_1.8.mat',
                 'ABLE7_091624_0.6.mat', 'ABLE7_091624_1.4.mat', 'ABLE7_091624_1.8.mat', 'ABLE7_091624_2.4.mat', 'ABLE8_111324_2.2.mat', 'ABLE8_111324_2.2_10deg.mat', 'ABLE8_111324_2.2_12deg.mat',
                 'ABLE8_111324_2.2_14deg.mat', 'ABLE8_111324_2.2_16deg.mat', 'ABLE8_111324_2.2_18deg.mat', 'ABLE8_111324_2.2_2deg.mat', 'ABLE8_111324_2.2_4deg.mat', 'ABLE8_111324_2.2_6deg.mat',
                 'ABLE8_111324_2.2_8deg.mat',
                'STROKE1_021925_0.5.mat', 'STROKE1_021925_0.5_1.0.mat', 'STROKE1_021925_0.75.mat', 'STROKE1_021925_1.0.mat', 'STROKE1_021925_1.0_2.mat',
                 'STROKE1_021925_Dyn.mat', 'STROKE1_021925_Dyn_2.mat'
                 ]
# balance_list = ['ABLE5_010325_ols_ab_r.mat', 'ABLE5_010325_ols_ab_r_2.mat', 'ABLE5_010325_ols_ab_r_3.mat', 'ABLE5_010325_ols_h_l.mat', 'ABLE5_010325_ols_h_r.mat',
#              'ABLE5_010325_ols_l.mat', 'ABLE5_010325_ols_l_2.mat', 'ABLE5_010325_ols_r.mat', 'ABLE5_010325_ols_r_2.mat', 'ABLE5_010325_roll_lr.mat',]

# Right Foot
# for fname in file_list:
#     if fname in ("ABLE1_112024_1.0.mat", "ABLE1_112024_0.75.mat", "ABLE1_112024_1.5.mat"):
#         height=1.7526
#         weight=77.11
#     if fname in ("ABLE2_030525_1.0.mat", "ABLE2_030525_0.75.mat", "ABLE2_030525_1.5.mat", "ABLE2_030525_2.0.mat"):
#         height=1.8288
#         weight=78.47
#     if fname in ("ABLE3_030525_1.0.mat", "ABLE3_030525_0.75.mat", "ABLE3_030525_1.5.mat", "ABLE3_030525_2.0.mat"):
#         height=1.778
#         weight=71.7
#     if fname in ("ABLE4_030525_1.0.mat", "ABLE4_030525_0.75.mat"):
#         height=1.6002
#         weight=70.3

subject_ids = []
balance_ids = []
walk_ids = []
# --- walk ---
all_insole = []
all_cop = []
for subj_id, fname in enumerate(file_list):
    path = os.path.join(file_dir, fname)
    norm_right_insole_all, norm_right_cop_insole = process_file(path)
    all_insole.append(norm_right_insole_all)
    all_cop.append(norm_right_cop_insole)

    n_steps = norm_right_insole_all.shape[-1]
    walk_ids.extend([subj_id] * n_steps)
    print(f"[{fname}] walk_insole max: {np.max(norm_right_insole_all):.2f}, min: {np.min(norm_right_insole_all):.2f}")

walk_insole = np.concatenate(all_insole, axis=3)
walk_cop = np.concatenate(all_cop, axis=2)
offset = len(file_list)

# --- balance ---
# all_insole = []  # reset
# for subj_id, fname in enumerate(balance_list):
#     path = os.path.join(file_dir, fname)
#     norm_right_insole_all = process_balance_file(path)
#     all_insole.append(norm_right_insole_all)
#
#     n_steps = norm_right_insole_all.shape[-1]
#     balance_ids.extend([subj_id + offset] * n_steps)
#     print(f"[{fname}] balance_insole max: {np.max(norm_right_insole_all):.2f}, min: {np.min(norm_right_insole_all):.2f}")
#
# balance_insole = np.concatenate(all_insole, axis=3)
# --- merge ---
# subject_ids = np.array(walk_ids + balance_ids)
# final_insole = np.concatenate([walk_insole, balance_insole], axis=3)

subject_ids = np.array(walk_ids)
final_insole = walk_insole
final_cop = walk_cop

print("final insole shape:", final_insole.shape)
print("final_cop.shape", final_cop)
np.savez(file_dir,
         subject_ids=subject_ids,
         final_insole=final_insole,
         final_cop=final_cop
         )