class ParallelAG_TFNN_Layer(nn.Module):
    def __init__(self, T, F, rT=12, rF=6, fusion_dim=256):
        super().__init__()
        self.T, self.F = T, F
        self.rT, self.rF = rT, rF
        self.fusion_dim = fusion_dim

        self.U_time    = nn.Parameter(torch.randn(T, rT) * 0.01)
        self.core_time = nn.Parameter(torch.randn(rT, F) * 0.01)
        self.U_freq    = nn.Parameter(torch.randn(F, rF) * 0.01)
        self.core_freq = nn.Parameter(torch.randn(T, rF) * 0.01)

        self.att_gate_time = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5,1), padding=(2,0)),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(96, 48, kernel_size=(7,1), padding=(3,0)),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(48, 1, kernel_size=(3,1), padding=(1,0)),
            nn.Sigmoid()
        )
        self.att_gate_freq = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(48, 24, kernel_size=(1,7), padding=(0,3)),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(24, 1, kernel_size=(1,3), padding=(0,1)),
            nn.Sigmoid()
        )

        # ── Weighted fusion components ─────────────────────────
        # Project each branch (different native sizes) into a shared space
        self.time_proj = nn.Linear(rT * F, fusion_dim)
        self.freq_proj = nn.Linear(T * rF, fusion_dim)

        # Learnable, trainable fusion weights (2 scalars -> softmax)
        self.fusion_weights = nn.Parameter(torch.ones(2))  # [w_time, w_freq]

    def forward(self, x, return_fusion_weights=False):
        batch_size = x.shape[0]
        if x.shape[1] != self.T or x.shape[2] != self.F:
            raise ValueError(
                f"Expected input shape (batch, {self.T}, {self.F}), got {x.shape}"
            )

        # ── Time branch ──────────────────────────────────────
        U_time_t           = self.U_time.T.unsqueeze(0).expand(batch_size, -1, -1)
        x_proj_time        = torch.bmm(U_time_t, x)
        core_time_exp      = self.core_time.unsqueeze(0).expand(batch_size, -1, -1)
        features_time      = (x_proj_time * core_time_exp) + x_proj_time
        features_time      = features_time.unsqueeze(1)
        gated_time         = self.att_gate_time(features_time)
        gated_features_time = (gated_time * features_time) + features_time

        # ── Frequency branch ─────────────────────────────────
        x_proj_freq        = torch.matmul(x, self.U_freq)
        core_freq_exp      = self.core_freq.unsqueeze(0).expand(batch_size, -1, -1)
        features_freq      = (x_proj_freq * core_freq_exp) + x_proj_freq
        features_freq      = features_freq.unsqueeze(1)
        gated_freq         = self.att_gate_freq(features_freq)
        gated_features_freq = (gated_freq * features_freq) + features_freq

        gated_features_freq = gated_features_freq.reshape(batch_size, -1)
        gated_features_time = gated_features_time.reshape(batch_size, -1)

        # ── Weighted fusion (replaces torch.cat) ───────────────
        t_proj = self.time_proj(gated_features_time)   # (B, fusion_dim)
        f_proj = self.freq_proj(gated_features_freq)    # (B, fusion_dim)

        w = torch.softmax(self.fusion_weights, dim=0)   # normalized, trainable weights
        fused = w[0] * t_proj + w[1] * f_proj            # (B, fusion_dim)

        if return_fusion_weights:
            return fused, w
        return fused


class ParallelAG_TFNN(nn.Module):
    def __init__(self, input_T, input_F, rT=12, rF=6, num_classes=2, fusion_dim=256):
        super().__init__()
        self.layer   = ParallelAG_TFNN_Layer(input_T, input_F, rT, rF, fusion_dim)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc      = nn.Linear(fusion_dim, num_classes)   # <-- was rT*input_F + input_T*rF

    def forward(self, x, return_features=False, before_dropout=False):
        features = self.layer(x)
        if before_dropout:
            if return_features:
                logits = self.fc(self.dropout(features))
                return logits, features
            return features
        features = self.dropout(features)
        logits   = self.fc(features)
        if return_features:
            return logits, features
        return logits
