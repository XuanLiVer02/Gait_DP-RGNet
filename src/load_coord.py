from scipy.io import loadmat
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coord_x = loadmat('C:/Users/XuanL/Documents/MATLAB/Gait Analysis/insole_coord.mat')['insole_coord_x']
coord_y = loadmat('C:/Users/XuanL/Documents/MATLAB/Gait Analysis/insole_coord.mat')['insole_coord_y']
mask = loadmat('C:/Users/XuanL/Documents/MATLAB/Gait Analysis/insole_coord.mat')['mask']

coord = np.stack((coord_x, coord_y), axis=2)
#   Flip
# coord[:32, :, :] = coord[:32, :, :][::-1, :, :]
coord = coord[:, ::-1, :]   #64, 16, 2

#####Normalization
y_coords = coord[:, :, 0]
x_coords = coord[:, :, 1]
H, W = 64, 16
y_coords_normalized = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min() + 1e-6)
x_coords_normalized = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-6)
coord = np.stack((x_coords_normalized, y_coords_normalized), axis=2)  # 64, 16, 2