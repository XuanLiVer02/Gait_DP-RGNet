import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTemporalRegressor(nn.Module):
    """
    temporal regression module
    Combine temporal prior and learned attention, then feed into LSTM for regression
    """

    def __init__(self,
                 pooled_nodes,
                 embed_dim,
                 lstm_hidden=64,
                 lstm_layers=2,
                 out_dim=6,
                 mode='attention_flatten',
                 bidirectional=True,
                 dropout=0.2,
                 target_attention_pattern=None,
                 prior_attention_weight=0.5,
                 learned_attention_temperature=0.5):
        super(GraphTemporalRegressor, self).__init__()

        self.attention_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        assert mode in ('flatten', 'mean', 'attention', 'attention_flatten')
        self.mode = mode
        self.Np = pooled_nodes
        self.D = embed_dim
        self.out_dim = out_dim
        self.bidirectional = bidirectional
        self.prior_attention_weight = prior_attention_weight
        self.learned_attention_temperature = learned_attention_temperature

        self.register_buffer('prior_attn', None)
        if target_attention_pattern is not None and self.prior_attention_weight > 0:
            if not isinstance(target_attention_pattern, torch.Tensor):
                target_attention_pattern = torch.tensor(target_attention_pattern, dtype=torch.float32)
            self.prior_attn = target_attention_pattern
            print(f"GraphTemporalRegressor: Prior Attention，Weight alpha = {self.prior_attention_weight}")

        self.last_attn_weights = None

        if mode == 'flatten' or mode == 'attention_flatten':
            self.input_dim_lstm = pooled_nodes * embed_dim
        elif mode == 'mean' or mode == 'attention':
            self.input_dim_lstm = embed_dim

        self.lstm = nn.LSTM(input_size=self.input_dim_lstm,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if lstm_layers > 1 else 0.0)

        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, out_dim)
        )

    def forward(self, X_pool_seq):
        B, T, Np, D = X_pool_seq.shape
        assert Np == self.Np and D == self.D

        final_attn_weights = None
        attn_weights_to_return = None
        x_in_raw = None
        learned_attn_weights = None

        if self.prior_attention_weight < 1.0:
            if self.mode in ('attention', 'attention_flatten'):
                attn_scores = self.attention_mlp(X_pool_seq)
                scaled_scores = attn_scores / self.learned_attention_temperature
                learned_attn_weights = F.softmax(scaled_scores, dim=2)

        if self.prior_attn is not None and self.prior_attention_weight > 0:
            prior_weights_expanded = self.prior_attn.unsqueeze(0).unsqueeze(-1).to(X_pool_seq.device)
            norm_sum = prior_weights_expanded.sum(dim=2, keepdim=True)
            prior_weights_expanded = prior_weights_expanded / (norm_sum + 1e-8)
            if learned_attn_weights is not None:
                final_attn_weights = (self.prior_attention_weight * prior_weights_expanded +
                                      (1.0 - self.prior_attention_weight) * learned_attn_weights)
            else:
                if self.mode in ('attention', 'attention_flatten'):
                    final_attn_weights = prior_weights_expanded
        elif learned_attn_weights is not None:
            final_attn_weights = learned_attn_weights

        if final_attn_weights is not None:
            attn_weights_to_return = final_attn_weights.squeeze(-1)
            self.last_attn_weights = attn_weights_to_return.detach()

        if self.mode == 'flatten':
            x_in_raw = X_pool_seq.reshape(B, T, Np * D)
        elif self.mode == 'mean':
            x_in_raw = X_pool_seq.mean(dim=2)
        elif self.mode == 'attention':
            if final_attn_weights is None: raise ValueError("Attention mode requires attention weights")
            weighted_features = X_pool_seq * final_attn_weights
            x_in_raw = torch.sum(weighted_features, dim=2)
        elif self.mode == 'attention_flatten':
            if final_attn_weights is None: raise ValueError("Attention_flatten mode requires attention weights")
            weighted_features = X_pool_seq * final_attn_weights
            x_in_raw = weighted_features.reshape(B, T, Np * D)

        lstm_input = x_in_raw
        lstm_out, _ = self.lstm(lstm_input)

        return lstm_out, attn_weights_to_return