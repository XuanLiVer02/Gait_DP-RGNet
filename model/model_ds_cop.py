import torch
import torch.nn as nn
import torch.nn.functional as F
from model_core import SpatialEncoder_HighRes, PositionalEncoder, LearnablePrototypePool, DynamicPositionalEncoder
from utils import GraphTemporalRegressor

class DynamicPrototypePooler_Decoupled(nn.Module):
    """
    Pre-LayerNorm: avoid outliers
    Learnable Prior: Each head learns how much to trust the partition map
    """
    def __init__(self, partition_map_64x16,
                 zero_coord_mask_64x16,
                 context_dim,
                 content_dim,
                 num_prototypes=6,
                 embed_dim=256,
                 num_heads=4,
                 prior_confidence=0.5):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Pre-LayerNorm
        self.ln_context = nn.LayerNorm(context_dim)
        self.ln_content = nn.LayerNorm(content_dim)

        # Attention Projections
        self.proto_queries = nn.Parameter(torch.randn(1, num_prototypes, embed_dim))
        self.to_k = nn.Linear(context_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(content_dim, embed_dim, bias=False)

        # Scaling factor (1 / sqrt(d_k))
        self.scale = self.head_dim ** -0.5

        # --- Physics Bias (Partition Map) ---
        partition_map_flat = partition_map_64x16.reshape(-1).long()
        bias_matrix = torch.zeros(num_prototypes, 64 * 16)

        # Mask: 1 - valid sensors in the correct prototype, 0 otherwise
        for k in range(num_prototypes):
            indices_k = (partition_map_flat == k).nonzero(as_tuple=True)[0]
            if len(indices_k) > 0:
                bias_matrix[k, indices_k] = 1.0

        self.register_buffer("physics_bias", bias_matrix.view(1, 1, num_prototypes, -1))    # Register buffer: (1, 1, Num_Proto, Num_Nodes)

        # Learnable Prior Scale (Per-Head)
        self.prior_scale = nn.Parameter(torch.full((num_heads, 1, 1), float(prior_confidence)))

        # Background Mask (0,0 coordinates)
        zero_coord_mask_flat = zero_coord_mask_64x16.reshape(-1)
        self.register_buffer("zero_coord_mask", zero_coord_mask_flat.view(1, 1, 1, -1))

        # visualization
        self.last_A_softmax = None

        # Initialization
        nn.init.normal_(self.proto_queries, std=0.02)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

    def forward(self, Z_context, Z_content):
        """
        Z_context: CNN spatial features (B*T, N, context_dim)
        Z_content: CoP embeddings (B*T, N, content_dim)
        """

        # 1. Pre-LayerNorm
        Z_context = self.ln_context(Z_context)
        Z_content = self.ln_content(Z_content)
        B_T, N, _ = Z_context.shape
        H = self.num_heads
        D_head = self.head_dim

        # 2. Linear Projections & Reshape
        # K, V: (B*T, N, H, D_h) -> (B*T, H, N, D_h)
        K = self.to_k(Z_context).view(B_T, N, H, D_head).transpose(1, 2)
        V = self.to_v(Z_content).view(B_T, N, H, D_head).transpose(1, 2)

        # Q: (1, Np, H*D_h) -> (B*T, H, Np, D_h)
        Q = self.proto_queries.expand(B_T, -1, -1).view(B_T, self.num_prototypes, H, D_head).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # (B*T, H, Np, D_h) @ (B*T, H, D_h, N) -> (B*T, H, Np, N)
        attn_logits = (Q @ K.transpose(-2, -1)) * self.scale

        # 4. Hard Masking (avoid invalid background nodes)
        attn_logits = attn_logits.masked_fill(self.zero_coord_mask, -1e9)

        # 5. Soft Biasing
        # Data_Logits + (Scale_Head_i * Partition_Map)
        # (B*T, H, Np, N) + (H, 1, 1) * (1, 1, Np, N)
        bias_term = self.physics_bias * self.prior_scale.unsqueeze(0)  # unsqueeze for Batch dim
        attn_with_bias = attn_logits + bias_term

        # 6. Softmax & Pooling
        A_softmax = attn_with_bias.softmax(dim=-1)
        self.last_A_softmax = A_softmax.mean(dim=1).detach()

        # Weighted Sum
        # (B*T, H, Np, N) @ (B*T, H, N, D_h) -> (B*T, H, Np, D_h)
        Z_pooled = A_softmax @ V

        # Concatenate Heads
        # (B*T, H, Np, D_h) -> (B*T, Np, H, D_h) -> (B*T, Np, Embed_Dim)
        Z_pooled = Z_pooled.transpose(1, 2).contiguous().view(B_T, self.num_prototypes, self.embed_dim)

        return Z_pooled, self.last_A_softmax

class DualPath_Hybrid_v3_Decoupled(nn.Module):
    """
    Main model
    """

    def __init__(self,
                 coord_64x16,
                 partition_map_64x16,
                 zero_coord_mask_64x16,
                 target_attention_pattern,
                 num_prototypes=6,
                 cnn_dim=128, pos_dim=128, embed_dim=256,
                 cop_embed_dim=256,
                 proto_lstm_hidden=256, bottleneck_dim=256, global_lstm_hidden=256,
                 lstm_layers=2, bidirectional=True, dropout=0.3, out_dim=9,
                 prior_confidence=0.5,
                 prior_attention_weight=0.5
                 ):
        super().__init__()
        self.num_nodes = 64 * 16
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim

        # 1. Shared Encoder
        self.cnn_encoder = SpatialEncoder_HighRes(embed_dim=cnn_dim)
        self.pos_encoder = PositionalEncoder(coord_64x16, embed_dim=pos_dim)
        self.cop_encoder = DynamicPositionalEncoder(in_dim=2, embed_dim=cop_embed_dim, num_freqs=8)

        # 2. Path A
        context_dim = pos_dim + cop_embed_dim + cnn_dim
        content_dim = cnn_dim

        self.pooler_prototype_A = DynamicPrototypePooler_Decoupled(
            partition_map_64x16=partition_map_64x16,
            zero_coord_mask_64x16=zero_coord_mask_64x16,
            context_dim=context_dim,
            content_dim=content_dim,
            num_prototypes=num_prototypes,
            embed_dim=embed_dim,
            num_heads=4,
            prior_confidence=prior_confidence
        )

        self.temporal_proto_A = GraphTemporalRegressor(
            pooled_nodes=num_prototypes, embed_dim=embed_dim,
            lstm_hidden=proto_lstm_hidden, lstm_layers=lstm_layers,
            out_dim=out_dim, mode='attention_flatten', bidirectional=bidirectional,
            dropout=dropout, target_attention_pattern=target_attention_pattern,
            prior_attention_weight=prior_attention_weight
        )
        proto_rnn_out_dim = proto_lstm_hidden * 2

        # 3. Path B
        path_b_input_dim = cnn_dim + pos_dim + cop_embed_dim
        self.kv_fusion_B = nn.Linear(path_b_input_dim, embed_dim)

        self.bottleneck_fc_B = nn.Linear(self.num_nodes * embed_dim, bottleneck_dim)
        self.temporal_global_B = nn.LSTM(
            input_size=bottleneck_dim, hidden_size=global_lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        global_rnn_out_dim = global_lstm_hidden * 2

        # 4. MLP head
        self.head_A = nn.Sequential(
            nn.LayerNorm(proto_rnn_out_dim), nn.Linear(proto_rnn_out_dim, 128),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, out_dim)
        )
        self.head_B = nn.Sequential(
            nn.LayerNorm(global_rnn_out_dim), nn.Linear(global_rnn_out_dim, 128),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, out_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, CoP):
        B, T, H, W = X.shape
        B_T = B * T

        # 1. Shared Features
        x_frames = X.view(B_T, 1, H, W)
        Z_feat_cnn = self.cnn_encoder(x_frames)
        Z_feat_pos = self.pos_encoder().expand(B_T, -1, -1)
        Z_feat_cop = self.cop_encoder(CoP.view(B_T, 2))
        Z_feat_cop_expanded = Z_feat_cop.unsqueeze(1).expand(-1, self.num_nodes, -1)

        # 2. Path A
        # K (Spatial Context + pos + CoP):
        Z_context_A = torch.cat([Z_feat_pos, Z_feat_cop_expanded, Z_feat_cnn], dim=-1)

        # V (Spatial Content)
        Z_content_A = Z_feat_cnn

        Z_pooled, _ = self.pooler_prototype_A(Z_context_A, Z_content_A)

        Z_seq_A = Z_pooled.view(B, T, self.num_prototypes, self.embed_dim)
        lstm_out_A, _ = self.temporal_proto_A(Z_seq_A)
        y_hat_A = self.head_A(lstm_out_A)

        # 3. Path B
        KV_input_B = torch.cat([Z_feat_cnn, Z_feat_pos, Z_feat_cop_expanded], dim=-1)
        KV_fused_B = F.relu(self.kv_fusion_B(KV_input_B))

        KV_fused_B = self.dropout(KV_fused_B)
        Z_flat_B = KV_fused_B.flatten(start_dim=1)
        Z_context_B = F.relu(self.bottleneck_fc_B(Z_flat_B))
        Z_seq_B = Z_context_B.view(B, T, -1)
        lstm_out_B, _ = self.temporal_global_B(Z_seq_B)
        y_hat_B = self.head_B(lstm_out_B)

        return y_hat_A + y_hat_B, None

    def get_last_spatial_attention(self):
        return self.pooler_prototype_A.last_A_softmax

    def get_last_temporal_attention(self):
        return self.temporal_proto_A.last_attn_weights

class Baseline_SpatialEncoder(nn.Module):
    """
    CNN without pooling
    """
    def __init__(self, H, W, frame_channels=1, embed_dim=128):  # (c_dim=128)
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
        self.num_nodes = H * W

    def forward(self, x):
        Z_cnn = self.encoder(x)  # (B*T, 128, 64, 16)
        B_T, D, H, W = Z_cnn.shape
        Z_nodes = Z_cnn.view(B_T, D, -1).permute(0, 2, 1)
        return Z_nodes  # (B*T, 1024, 128)

class Baseline_PlainCNN_6DOF(nn.Module):
    def __init__(self, coord_map, out_dim=6, dropout=0.3):
        super().__init__()

        H, W, _ = coord_map.shape
        self.num_nodes = H * W  # (1024)

        cnn_dim = 128
        pos_dim = 128
        embed_dim = 256
        cop_embed_dim = 256
        bottleneck_dim = 256

        self.cnn_encoder = Baseline_SpatialEncoder(H, W, embed_dim=cnn_dim)
        self.pos_encoder = PositionalEncoder(coord_map, embed_dim=pos_dim)
        self.cop_encoder = DynamicPositionalEncoder(in_dim=2, embed_dim=cop_embed_dim)

        path_b_input_dim = cnn_dim + pos_dim + cop_embed_dim  # 512
        self.kv_fusion_B = nn.Linear(path_b_input_dim, embed_dim)  # (in: 512, out: 256)
        self.bottleneck_fc_B = nn.Linear(self.num_nodes * embed_dim, bottleneck_dim)  # (in: 1024*256)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)  # (out: 6)
        )

    def forward(self, X, CoP):
        B, T, H, W = X.shape
        B_T = B * T
        x_frames = X.view(B_T, 1, H, W)

        Z_feat_cnn = self.cnn_encoder(x_frames)  # (B*T, 1024, 128)
        Z_feat_pos = self.pos_encoder().expand(B_T, -1, -1)  # (B*T, 1024, 128)
        Z_feat_cop = self.cop_encoder(CoP.view(B_T, 2))  # (B*T, 256)
        Z_feat_cop_expanded = Z_feat_cop.unsqueeze(1).expand(-1, self.num_nodes, -1)  # (B*T, 1024, 256)

        KV_input_B = torch.cat([Z_feat_cnn, Z_feat_pos, Z_feat_cop_expanded], dim=-1)
        KV_fused_B = F.relu(self.kv_fusion_B(KV_input_B))  # (B*T, 1024, 256)
        KV_fused_B = self.dropout(KV_fused_B)

        Z_flat_B = KV_fused_B.flatten(start_dim=1)  # (B*T, 1024*256)
        Z_context_B = F.relu(self.bottleneck_fc_B(Z_flat_B))  # (B*T, 256)

        output = self.head(Z_context_B)  # (B*T, 6)
        return output.view(B, T, -1), None  # (B, T, 6)

class Baseline_CNN_LSTM_6DOF(nn.Module):
    def __init__(self, coord_map, out_dim=6, global_lstm_hidden=256,
                 lstm_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()

        H, W, _ = coord_map.shape
        self.num_nodes = H * W  # (1024)

        cnn_dim = 128
        pos_dim = 128
        embed_dim = 256
        cop_embed_dim = 256
        bottleneck_dim = 256

        self.cnn_encoder = Baseline_SpatialEncoder(H, W, embed_dim=cnn_dim)
        self.pos_encoder = PositionalEncoder(coord_map, embed_dim=pos_dim)
        self.cop_encoder = DynamicPositionalEncoder(in_dim=2, embed_dim=cop_embed_dim)

        path_b_input_dim = cnn_dim + pos_dim + cop_embed_dim  # 512
        self.kv_fusion_B = nn.Linear(path_b_input_dim, embed_dim)  # (in: 512, out: 256)
        self.bottleneck_fc_B = nn.Linear(self.num_nodes * embed_dim, bottleneck_dim)

        self.temporal_global_B = nn.LSTM(
            input_size=bottleneck_dim,  # (256)
            hidden_size=global_lstm_hidden,  # (256)
            num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        global_rnn_out_dim = global_lstm_hidden * 2  # 512

        self.head_B = nn.Sequential(
            nn.LayerNorm(global_rnn_out_dim),
            nn.Linear(global_rnn_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)  # (out: 6)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, CoP):
        B, T, H, W = X.shape
        B_T = B * T
        x_frames = X.view(B_T, 1, H, W)

        Z_feat_cnn = self.cnn_encoder(x_frames)  # (B*T, 1024, 128)
        Z_feat_pos = self.pos_encoder().expand(B_T, -1, -1)  # (B*T, 1024, 128)
        Z_feat_cop = self.cop_encoder(CoP.view(B_T, 2))  # (B*T, 256)
        Z_feat_cop_expanded = Z_feat_cop.unsqueeze(1).expand(-1, self.num_nodes, -1)  # (B*T, 1024, 256)

        KV_input_B = torch.cat([Z_feat_cnn, Z_feat_pos, Z_feat_cop_expanded], dim=-1)
        KV_fused_B = F.relu(self.kv_fusion_B(KV_input_B))  # (B*T, 1024, 256)
        KV_fused_B = self.dropout(KV_fused_B)

        Z_flat_B = KV_fused_B.flatten(start_dim=1)  # (B*T, 1024*256)
        Z_context_B = F.relu(self.bottleneck_fc_B(Z_flat_B))  # (B*T, 256)

        Z_seq_B = Z_context_B.view(B, T, -1)
        lstm_out_B, _ = self.temporal_global_B(Z_seq_B)  # (B, T, 512)

        output = self.head_B(lstm_out_B)  # (B, T, 6)
        return output, None

class DualPath_Hybrid_v3_PathBOnly(nn.Module):
    """
    Path-B-Only Baseline
    - Shared (CNN + coord + CoP)
    - Path B (bottleneck + LSTM + head_B)
    - Remove Path A
    """

    def __init__(self,
                 coord_64x16,
                 cnn_dim=128, pos_dim=128, embed_dim=256,
                 cop_embed_dim=256,
                 bottleneck_dim=256, global_lstm_hidden=256,
                 lstm_layers=2, bidirectional=True, dropout=0.3, out_dim=9,
                 ):
        super().__init__()
        self.num_nodes = 64 * 16
        self.embed_dim = embed_dim

        # 1. Shared Encoder
        self.cnn_encoder = SpatialEncoder_HighRes(embed_dim=cnn_dim)
        self.pos_encoder = PositionalEncoder(coord_64x16, embed_dim=pos_dim)
        self.cop_encoder = DynamicPositionalEncoder(in_dim=2, embed_dim=cop_embed_dim, num_freqs=8)

        # 2. Path B
        path_b_input_dim = cnn_dim + pos_dim + cop_embed_dim
        self.kv_fusion_B = nn.Linear(path_b_input_dim, embed_dim)

        self.bottleneck_fc_B = nn.Linear(self.num_nodes * embed_dim, bottleneck_dim)
        self.temporal_global_B = nn.LSTM(
            input_size=bottleneck_dim, hidden_size=global_lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout
        )
        global_rnn_out_dim = global_lstm_hidden * 2

        # 3. Regression Head
        self.head_B = nn.Sequential(
            nn.LayerNorm(global_rnn_out_dim),
            nn.Linear(global_rnn_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, CoP):
        B, T, H, W = X.shape
        B_T = B * T

        # 1. Shared Feature Extraction
        x_frames = X.view(B_T, 1, H, W)
        Z_feat_cnn = self.cnn_encoder(x_frames)                      # (B*T, N, cnn_dim)
        Z_feat_pos = self.pos_encoder().expand(B_T, -1, -1)          # (B*T, N, pos_dim)
        Z_feat_cop = self.cop_encoder(CoP.view(B_T, 2))              # (B*T, cop_dim)
        Z_feat_cop_expanded = Z_feat_cop.unsqueeze(1).expand(
            -1, self.num_nodes, -1
        )                                                             # (B*T, N, cop_dim)

        # 2. Path B
        KV_input_B = torch.cat([Z_feat_cnn, Z_feat_pos, Z_feat_cop_expanded], dim=-1)
        KV_fused_B = F.relu(self.kv_fusion_B(KV_input_B))

        KV_fused_B = self.dropout(KV_fused_B)
        Z_flat_B = KV_fused_B.flatten(start_dim=1)                   # (B*T, N*embed_dim)
        Z_context_B = F.relu(self.bottleneck_fc_B(Z_flat_B))         # (B*T, bottleneck_dim)

        Z_seq_B = Z_context_B.view(B, T, -1)                         # (B, T, bottleneck_dim)
        lstm_out_B, _ = self.temporal_global_B(Z_seq_B)              # (B, T, 2*hidden)

        y_hat_B = self.head_B(lstm_out_B)                            # (B, T, out_dim)

        return y_hat_B, None

    def get_last_spatial_attention(self):
        return None

    def get_last_temporal_attention(self):
        return None