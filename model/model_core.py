import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialEncoder_HighRes(nn.Module):
    def __init__(self, frame_channels=1, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(frame_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(),
        )
        self.output_dim = embed_dim
        self.num_nodes = 64 * 16  # 1024

    def forward(self, x):
        # x: (B*T, 1, 64, 16)
        Z_cnn = self.encoder(x)  # (B*T, D, 64, 16)
        B_T, D, H, W = Z_cnn.shape
        # (B*T, D, 64, 16) -> (B*T, 1024, D)
        Z_nodes = Z_cnn.view(B_T, D, -1).permute(0, 2, 1)
        return Z_nodes

class PositionalEncoder(nn.Module):
    """ Encode (x, y) coordinates into embeddings using Fourier features + MLP."""

    def __init__(self, coord_data_64x16, embed_dim=128, num_freqs=8):
        super().__init__()

        device = coord_data_64x16.device
        coords_flat = coord_data_64x16.reshape(-1, 2).float()
        min_val, _ = torch.min(coords_flat, dim=0, keepdim=True)
        max_val, _ = torch.max(coords_flat, dim=0, keepdim=True)
        coords_norm = 2.0 * (coords_flat - min_val) / (max_val - min_val + 1e-8) - 1.0

        freq_bands = 2.0 ** torch.arange(num_freqs, device=device)
        freq_inputs = coords_norm.unsqueeze(-1) * freq_bands.unsqueeze(0).unsqueeze(0)
        periodic_inputs = torch.cat([torch.sin(freq_inputs), torch.cos(freq_inputs)], dim=-1)
        periodic_inputs = periodic_inputs.view(1024, -1)
        fourier_dim = periodic_inputs.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim * 2), nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.mlp.to(device)

        pos_encoding_flat = self.mlp(periodic_inputs)
        self.register_buffer('pos_encoding', pos_encoding_flat.detach().unsqueeze(0))

    def forward(self):
        return self.pos_encoding

# For COP
class DynamicPositionalEncoder(nn.Module):
    """
    dynamic positional encoder COP
    (B, T, 2) -> (B, T, embed_dim)
    """

    def __init__(self, in_dim=2, embed_dim=256, num_freqs=8):
        super().__init__()
        self.num_freqs = num_freqs
        self.register_buffer('freq_bands', 2.0 ** torch.arange(num_freqs))

        fourier_dim = in_dim * 2 * num_freqs  # 2 * 2 * 8 = 32

        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, coords):
        """
        coords: (B, T, 2)
        """
        freq_inputs = coords.unsqueeze(-1) * self.freq_bands    #(B, T, 2, 1) * (8,) -> (B, T, 2, 8)

        # cat sin/cos -> (..., 2 * 2 * num_freqs)
        periodic_inputs = torch.cat([torch.sin(freq_inputs), torch.cos(freq_inputs)], dim=-1)

        # flatten
        input_shape = coords.shape[:-1]  # (B, T)
        periodic_flat = periodic_inputs.view(*input_shape, -1)  # (B, T, fourier_dim)

        encoding = self.mlp(periodic_flat)  # (B, T, embed_dim)

        return encoding

class Encoder_With_Pos(nn.Module):
    """
    Concatenate CNN spatial features + positional embeddings
    """
    def __init__(self, coord_64x16, cnn_dim=128, pos_dim=128, embed_dim=256):
        super().__init__()
        self.cnn_encoder = SpatialEncoder_HighRes(embed_dim=cnn_dim)
        self.pos_encoder = PositionalEncoder(coord_64x16, embed_dim=pos_dim)
        self.kv_fusion = nn.Linear(cnn_dim + pos_dim, embed_dim)
        self.num_nodes = 1024

    def forward(self, x_frames):
        B_T = x_frames.shape[0]

        Z_feat_cnn = self.cnn_encoder(x_frames)  # (B*T, 1024, cnn_dim)
        Z_feat_pos = self.pos_encoder()  # (1, 1024, pos_dim)

        KV_input = torch.cat([Z_feat_cnn, Z_feat_pos.expand(B_T, -1, -1)], dim=-1)
        KV_fused = F.relu(self.kv_fusion(KV_input))  # (B*T, 1024, embed_dim)

        return KV_fused  # (B*T, 1024, D_model)

class LearnablePrototypePool(nn.Module):
    def __init__(self, partition_map_64x16, num_prototypes=6,
                 high_confidence=10.0, low_confidence=0.0):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_nodes = 64 * 16
        device = partition_map_64x16.device
        partition_map_flat = partition_map_64x16.reshape(-1).long()
        prior_logits = torch.full(
            (self.num_nodes, num_prototypes),
            low_confidence,
            device=device
        )
        valid_sensor_indices = torch.where(partition_map_flat >= 0)[0]
        prototype_indices_for_valid_sensors = partition_map_flat[valid_sensor_indices]
        prior_logits[valid_sensor_indices, prototype_indices_for_valid_sensors] = high_confidence
        self.assignment_logits = nn.Parameter(prior_logits)
        self.softmax = nn.Softmax(dim=1)

        # For visualization
        self.last_A_softmax = None

    def forward(self, Z_kv):
        B_T, N, D = Z_kv.shape

        A_softmax = self.softmax(self.assignment_logits)
        self.last_A_softmax = A_softmax.detach()

        A_T = A_softmax.transpose(0, 1)
        A_T_batch = A_T.unsqueeze(0).expand(B_T, -1, -1)
        Z_prototypes = torch.bmm(A_T_batch, Z_kv)

        return Z_prototypes, A_softmax