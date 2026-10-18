import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Helper: Lossless 2D Haar Wavelet Transform (DWT / IDWT) in pure PyTorch
# ============================================================================
def haar_dwt2d(x: torch.Tensor):
    """
    Single-level 2D Haar DWT.
    Args:
        x: [B, C, H, W] (pads by 1 if H or W is odd)
    Returns:
        LL, LH, HL, HH each of shape [B, C, H_pad//2, W_pad//2], and (H, W)
    """
    B, C, H, W = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) * 0.5
    LH = (x00 + x01 - x10 - x11) * 0.5  # Horizontal edges
    HL = (x00 - x01 + x10 - x11) * 0.5  # Vertical edges
    HH = (x00 - x01 - x10 + x11) * 0.5  # Diagonal edges
    return LL, LH, HL, HH, (H, W)


def haar_idwt2d(LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor, orig_size):
    """
    Single-level 2D Haar Inverse DWT.
    """
    B, C, Hh, Wh = LL.shape
    x00 = (LL + LH + HL + HH) * 0.5
    x01 = (LL + LH - HL - HH) * 0.5
    x10 = (LL - LH + HL - HH) * 0.5
    x11 = (LL - LH - HL + HH) * 0.5

    out = torch.zeros((B, C, Hh * 2, Wh * 2), dtype=LL.dtype, device=LL.device)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    H, W = orig_size
    return out[:, :, :H, :W]


# ============================================================================
# Innovation 1: Multi-Prototype Normal Sub-Cluster Bank & Background Suppression
# (Inspired by PGBL CVPR'26 + ArcAD ECCV'26 + FB-CLIP CVPR'26)
# ============================================================================
class MultiPrototypeBackgroundDisentangler(nn.Module):
    """
    1. Expands the single global normal token into M semantic sub-cluster prototypes
       to capture multi-modal normal bridge backgrounds (concrete pores, joints, shadows).
    2. Extracts a per-image background prototype from high-normality tokens and subtracts
       background shift to eliminate cross-bridge lighting/material variance.
    """
    def __init__(self, embed_dim: int = 1024, num_layers: int = 4, num_prototypes: int = 6, temp: float = 0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_prototypes = num_prototypes
        self.temp = temp

        # Orthogonal-initialized prototype offsets per layer: [num_layers, M, C]
        offsets = torch.randn(num_layers, num_prototypes, embed_dim) * 0.02
        for l in range(num_layers):
            nn.init.orthogonal_(offsets[l])
        self.proto_offsets = nn.Parameter(offsets)
        self.proto_scale = nn.Parameter(torch.full((num_layers, 1, 1), 0.1))

        # Per-layer background suppression projection & learnable gate (initialized small)
        self.bg_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, embed_dim),
            ) for _ in range(num_layers)
        ])
        for m in self.bg_proj:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
        self.bg_gate = nn.Parameter(torch.full((num_layers,), 0.1))

    def get_normal_prototypes(self, normal_token: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Args:
            normal_token: [B, C]
        Returns:
            normal_protos: [B, M, C] L2-normalized multi-prototype normal bank
        """
        base = F.normalize(normal_token, dim=-1, eps=1e-8).unsqueeze(1)  # [B, 1, C]
        offset = F.normalize(self.proto_offsets[layer_idx], dim=-1, eps=1e-8).unsqueeze(0)  # [1, M, C]
        scale = torch.clamp(self.proto_scale[layer_idx], 0.01, 0.5)
        protos = F.normalize(base + scale * offset, dim=-1, eps=1e-8)  # [B, M, C]
        return protos

    def suppress_background(self, patch_tokens: torch.Tensor, anomaly_token: torch.Tensor,
                            normal_token: torch.Tensor, layer_idx: int):
        """
        Args:
            patch_tokens: [B, N, C]
            anomaly_token: [B, C]
            normal_token: [B, C]
        Returns:
            suppressed_patches: [B, N, C]
            anomaly_map_layer: [B, H_p, W_p]
        """
        B, N, C = patch_tokens.shape
        H_p = int(math.isqrt(N))

        anom_norm = F.normalize(anomaly_token, dim=-1, eps=1e-8)  # [B, C]
        norm_protos = self.get_normal_prototypes(normal_token, layer_idx)  # [B, M, C]
        patches_norm = F.normalize(patch_tokens, dim=-1, eps=1e-8)  # [B, N, C]

        # Cosine similarity to anomaly token: [B, N]
        sim_anom = torch.bmm(patches_norm, anom_norm.unsqueeze(-1)).squeeze(-1)

        # Cosine similarity to M normal prototypes: [B, N, M]
        sim_norm_all = torch.bmm(patches_norm, norm_protos.transpose(1, 2))
        # Smooth max across M normal sub-clusters
        sim_norm = self.temp * torch.logsumexp(sim_norm_all / self.temp, dim=-1) - self.temp * math.log(self.num_prototypes)

        # Identify top 50% most normal patches per image as background bank (FB-CLIP BS)
        init_normality = (sim_norm - sim_anom).detach()  # [B, N]
        k_bg = max(1, N // 2)
        bg_indices = torch.topk(init_normality, k=k_bg, dim=-1).indices  # [B, k_bg]
        bg_indices_exp = bg_indices.unsqueeze(-1).expand(-1, -1, C)
        bg_tokens = torch.gather(patch_tokens, dim=1, index=bg_indices_exp)  # [B, k_bg, C]

        # Prototypical background vector: 0.5 * Mean + 0.5 * Max
        b_proto = 0.5 * bg_tokens.mean(dim=1) + 0.5 * bg_tokens.max(dim=1).values  # [B, C]
        bg_shift = self.bg_proj[layer_idx](b_proto).unsqueeze(1)  # [B, 1, C]

        # Similarity-weighted background subtraction
        s_bg = F.cosine_similarity(patch_tokens, b_proto.unsqueeze(1), dim=-1).clamp(min=0.0).unsqueeze(-1)  # [B, N, 1]
        gate = torch.tanh(self.bg_gate[layer_idx])
        suppressed_patches = patch_tokens - gate * (1.0 - s_bg * 0.5) * bg_shift

        # Recompute refined similarity after background suppression
        supp_norm = F.normalize(suppressed_patches, dim=-1, eps=1e-8)
        sim_anom_final = torch.bmm(supp_norm, anom_norm.unsqueeze(-1)).squeeze(-1)
        sim_norm_final_all = torch.bmm(supp_norm, norm_protos.transpose(1, 2))
        sim_norm_final = self.temp * torch.logsumexp(sim_norm_final_all / self.temp, dim=-1) - self.temp * math.log(self.num_prototypes)

        score_patch = sim_anom_final - sim_norm_final  # [B, N]
        anomaly_map_layer = score_patch.view(B, H_p, H_p)
        return suppressed_patches, anomaly_map_layer

    def diversity_loss(self) -> torch.Tensor:
        """
        Penalize excessive similarity between distinct normal sub-cluster prototype offsets (PGBL Eq. 5).
        """
        losses = []
        for l in range(self.num_layers):
            p = F.normalize(self.proto_offsets[l], dim=-1, eps=1e-8)  # [M, C]
            sim_mat = torch.mm(p, p.t())  # [M, M]
            mask = ~torch.eye(self.num_prototypes, dtype=torch.bool, device=p.device)
            off_diag = sim_mat[mask].view(self.num_prototypes, self.num_prototypes - 1)
            # Encourage cosine similarity between distinct offsets to be <= 0.2
            losses.append(F.relu(off_diag.max(dim=-1).values - 0.2).mean())
        return torch.stack(losses).mean()


# ============================================================================
# Innovation 2: Wavelet-Directional Self-Alignment Function (WD-SAF)
# (Inspired by DLVP-CLIP CVPR'26 + WPFormer CVPR'25 + MixerCSeg CVPR'26)
# ============================================================================
class WaveletDirectionalSAF(nn.Module):
    """
    Replaces VisualAD's point-wise MLP SAF with a 3-branch module:
      1. Point-wise MLP branch (preserves original VisualAD SAF capacity)
      2. 2D Haar Wavelet High-Frequency MSCM branch (counteracts ViT low-pass filtering)
      3. Directional Strip-Convolution Edge Gate (captures elongated crack topology)
    """
    def __init__(self, embed_dim: int = 1024, bottleneck_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim

        # Branch 1: Original point-wise 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Shared bottleneck down/up projections for spatial-frequency branches
        self.down_proj = nn.Conv2d(embed_dim, bottleneck_dim, kernel_size=1, bias=False)
        self.up_proj = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        # Branch 2: Wavelet High-Frequency MSCM modulation (WPFormer + DLVP-CLIP)
        self.wav_global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(bottleneck_dim, bottleneck_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim // 4, bottleneck_dim, kernel_size=1),
        )
        self.wav_local_gate = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim // 4, kernel_size=1),
            nn.GroupNorm(8, bottleneck_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim // 4, bottleneck_dim, kernel_size=1),
        )

        # Branch 3: Directional Strip-Convolution Edge Gate (MixerCSeg DEGConv)
        self.strip_h = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=(1, 5), padding=(0, 2), groups=bottleneck_dim)
        self.strip_v = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=(5, 1), padding=(2, 0), groups=bottleneck_dim)
        self.strip_fuse = nn.Sequential(
            nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, bottleneck_dim),
            nn.GELU(),
        )
        self.edge_gate = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, groups=bottleneck_dim),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, N, C] where N = H_p * W_p (e.g., 37*37 = 1369)
        Returns:
            recalibrated_tokens: [B, N, C]
        """
        B, N, C = patch_tokens.shape
        out_mlp = self.mlp(patch_tokens.reshape(-1, C)).view(B, N, C)

        H_p = int(math.isqrt(N))
        if H_p * H_p != N:
            return out_mlp

        # Reshape to 2D spatial grid [B, C, H_p, W_p]
        feat_2d = patch_tokens.transpose(1, 2).contiguous().view(B, C, H_p, H_p)
        z = self.down_proj(feat_2d)  # [B, Cb, H_p, W_p]

        # Wavelet branch
        LL, LH, HL, HH, orig_hw = haar_dwt2d(z)
        HF = LH + HL + HH
        sum_freq = LL + HF
        w_freq = torch.sigmoid(self.wav_global_gate(sum_freq) + self.wav_local_gate(sum_freq))
        LH_mod = LH * w_freq
        HL_mod = HL * w_freq
        HH_mod = HH * w_freq
        z_wav = haar_idwt2d(LL, LH_mod, HL_mod, HH_mod, orig_hw)

        # Directional strip-conv branch
        sh = self.strip_h(z)
        sv = self.strip_v(z)
        s_dir = self.strip_fuse(torch.cat([sh, sv], dim=1))
        z_strip = s_dir * self.edge_gate(z)

        z_combined = self.up_proj(z_wav + z_strip)  # [B, C, H_p, W_p]
        spatial_residual = z_combined.flatten(2).transpose(1, 2)  # [B, N, C]
        return out_mlp + self.res_scale * spatial_residual


# ============================================================================
# Innovation 3: Anomaly-Guided Dynamic Scanning & Wavelet-Gated High-Res Decoder
# (Inspired by CrackSSM CVPR'26 + SCSegamba CVPR'25)
# ============================================================================
def _build_serpentine_paths(H: int, W: int):
    """
    Precompute 4 continuous serpentine scan index permutations of length H*W (SCSegamba SASS):
      Path 0: Horizontal serpentine
      Path 1: Vertical serpentine
      Path 2: Main-diagonal serpentine (r + c)
      Path 3: Anti-diagonal serpentine (r - c)
    """
    grid = torch.arange(H * W).view(H, W)
    # Path 0: Horizontal snake
    p0 = []
    for r in range(H):
        row = grid[r].tolist()
        p0.extend(row if r % 2 == 0 else row[::-1])
    # Path 1: Vertical snake
    p1 = []
    for c in range(W):
        col = grid[:, c].tolist()
        p1.extend(col if c % 2 == 0 else col[::-1])
    # Path 2: Main diagonal snake
    diags2 = {}
    for r in range(H):
        for c in range(W):
            d = r + c
            diags2.setdefault(d, []).append(int(grid[r, c]))
    p2 = []
    for d in sorted(diags2.keys()):
        seg = diags2[d]
        p2.extend(seg if d % 2 == 0 else seg[::-1])
    # Path 3: Anti-diagonal snake
    diags3 = {}
    for r in range(H):
        for c in range(W):
            d = r - c
            diags3.setdefault(d, []).append(int(grid[r, c]))
    p3 = []
    for i, d in enumerate(sorted(diags3.keys())):
        seg = diags3[d]
        p3.extend(seg if i % 2 == 0 else seg[::-1])

    paths = torch.tensor([p0, p1, p2, p3], dtype=torch.long)  # [4, N]
    return paths


class AnomalyGuidedDynamicScanDecoder(nn.Module):
    """
    1. Reorders patch tokens along 4 serpentine routes using the coarse anomaly map as key
       (CrackSSM ATR + SCSegamba SASS) so disconnected crack segments interact directly in 1D.
    2. Refines the upsampled anomaly map at full resolution (518x518) using single-level
       Haar wavelet high-frequency sub-bands from the raw input image (CrackSSM WGM + GRO),
       and outputs both the refined anomaly map and a pixel-wise log-confidence map (AnomalyVFM).
    """
    def __init__(self, embed_dim: int = 1024, num_layers: int = 4, patch_grid: int = 37, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.patch_grid = patch_grid
        self.hidden_dim = hidden_dim

        paths = _build_serpentine_paths(patch_grid, patch_grid)
        self.register_buffer("scan_paths", paths, persistent=False)  # [4, N]

        self.token_reduce = nn.Linear(embed_dim, hidden_dim)
        # 1D sequence mixer along each reordered scan path (emulating Mamba S6 local-global state transition)
        self.seq_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2, groups=hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.Sigmoid(),
            ) for _ in range(4)
        ])
        self.path_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scan_head = nn.Sequential(
            nn.Conv2d(hidden_dim * 4 + num_layers, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.scan_head[-1].weight)
        nn.init.zeros_(self.scan_head[-1].bias)

        # Wavelet Gating Module (WGM) & Global Re-Optimization (GRO) on raw image Haar sub-bands
        self.wgm_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Sigmoid(),
            ) for _ in range(3)  # LH, HL, HH
        ])
        self.highres_refine = nn.Sequential(
            nn.Conv2d(num_layers + 4, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1),  # Channel 0: residual score delta, Channel 1: log-confidence c
        )
        nn.init.zeros_(self.highres_refine[-1].weight)
        nn.init.zeros_(self.highres_refine[-1].bias)
        self.scan_scale = nn.Parameter(torch.tensor(0.2))
        self.wav_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, raw_image: torch.Tensor, layer_patch_maps: list, top_patch_tokens: torch.Tensor, image_size: int):
        """
        Args:
            raw_image: [B, 3, H, W] normalized input image
            layer_patch_maps: List of 4 tensors each [B, H_p, W_p] (before upsampling)
            top_patch_tokens: [B, N, C] recalibrated patch tokens
            image_size: target output resolution (518)
        Returns:
            final_anomaly_map: [B, H, W]
            layer_upsampled_maps: List of 4 tensors [B, H, W] (including scan + wavelet refinement)
            log_confidence_map: [B, H, W]
        """
        B, N, C = top_patch_tokens.shape
        H_p = int(math.isqrt(N))

        stacked_lowres = torch.stack(layer_patch_maps, dim=1)  # [B, L, H_p, W_p]
        coarse_map = stacked_lowres.sum(dim=1)  # [B, H_p, W_p]
        coarse_flat = coarse_map.view(B, N)

        # Reduce token dim for fast dynamic scanning
        x_tok = self.token_reduce(top_patch_tokens)  # [B, N, D_h]

        path_outputs = []
        paths = self.scan_paths
        if paths.shape[1] != N:
            paths = _build_serpentine_paths(H_p, H_p).to(top_patch_tokens.device)

        # Anomaly-Guided Dynamic Scanning along 4 serpentine routes
        for k in range(4):
            p_idx = paths[k].unsqueeze(0).expand(B, -1)  # [B, N]
            scores_k = torch.gather(coarse_flat, dim=1, index=p_idx)  # [B, N]
            # Quantize + sort by anomaly intensity so anomalous tokens are grouped together (CrackSSM ATR)
            sort_order = torch.argsort(scores_k, dim=1, descending=True)  # [B, N]
            total_perm = torch.gather(p_idx, dim=1, index=sort_order)  # [B, N]

            perm_exp = total_perm.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            x_sorted = torch.gather(x_tok, dim=1, index=perm_exp)  # [B, N, D_h]

            # 1D sequence mixing
            x_seq = x_sorted.transpose(1, 2)  # [B, D_h, N]
            gate_seq = self.seq_mixers[k](x_seq)
            x_mixed = (x_seq * gate_seq).transpose(1, 2)  # [B, N, D_h]

            # Un-sort back to original 2D spatial indices
            unsort_full = torch.argsort(total_perm, dim=1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            x_restored = torch.gather(x_mixed, dim=1, index=unsort_full)  # [B, N, D_h]
            path_outputs.append(x_restored.transpose(1, 2).contiguous().view(B, self.hidden_dim, H_p, H_p))

        cat_paths = torch.cat(path_outputs, dim=1)  # [B, 4*D_h, H_p, W_p]
        cat_paths = cat_paths * self.path_se(cat_paths)
        scan_delta_lowres = self.scan_head(torch.cat([cat_paths, stacked_lowres], dim=1)).squeeze(1)  # [B, H_p, W_p]

        # Upsample each layer map + scan delta to full resolution
        layer_upsampled = []
        for l_map in layer_patch_maps:
            enhanced_l = l_map + torch.tanh(self.scan_scale) * scan_delta_lowres
            up_l = F.interpolate(
                enhanced_l.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False
            ).squeeze(1)
            layer_upsampled.append(up_l)

        base_highres_sum = torch.stack(layer_upsampled, dim=0).sum(dim=0)  # [B, H, W]

        # Raw-Image Haar Wavelet-Gated High-Res Refinement (CrackSSM WGM)
        _, LH_img, HL_img, HH_img, _ = haar_dwt2d(raw_image)
        hf_bands = [LH_img.abs().mean(dim=1, keepdim=True),
                    HL_img.abs().mean(dim=1, keepdim=True),
                    HH_img.abs().mean(dim=1, keepdim=True)]
        gated_maps = []
        base_unsq = base_highres_sum.unsqueeze(1)  # [B, 1, H, W]
        for b_idx, hf in enumerate(hf_bands):
            w_g = self.wgm_gates[b_idx](hf)
            w_g_up = F.interpolate(w_g, size=(image_size, image_size), mode="bilinear", align_corners=False)
            gated_maps.append(base_unsq * w_g_up)

        highres_in = torch.cat(
            [torch.stack(layer_upsampled, dim=1), base_unsq] + gated_maps, dim=1
        )  # [B, L + 4, H, W]
        ref_out = self.highres_refine(highres_in)  # [B, 2, H, W]
        highres_delta = ref_out[:, 0, :, :]
        log_conf = ref_out[:, 1, :, :]

        w_scale = torch.tanh(self.wav_scale)
        final_anomaly_map = base_highres_sum + w_scale * highres_delta
        # Also add 1/L of the highres wavelet delta to each layer map for deep supervision consistency
        layer_upsampled_refined = [m + (w_scale / len(layer_upsampled)) * highres_delta for m in layer_upsampled]

        return final_anomaly_map, layer_upsampled_refined, log_conf


# ============================================================================
# Innovation 4: Uncertainty-Aware Edge Loss & Dynamic Area-Aware Top-p% Scorer
# (Inspired by AnomalyVFM CVPR'26 + DCP-SFR CVPR'26)
# ============================================================================
class DynamicAreaScorer(nn.Module):
    """
    Computes image-level anomaly score by dynamically weighting multiple Top-p% pooling ratios
    (p in {0.2%, 1.0%, 5.0%}) conditioned on anomaly map response statistics, so both hairline
    cracks (<0.3% pixels) and large spalling (>15% pixels) are accurately scored.
    """
    def __init__(self, ratios=(0.002, 0.01, 0.05)):
        super().__init__()
        self.ratios = ratios
        self.gate_mlp = nn.Sequential(
            nn.Linear(len(ratios) + 2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, len(ratios)),
        )
        # Initialize bias to favor 1.0% (index 1) at start of training, matching VisualAD default
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, 0.0)
        with torch.no_grad():
            self.gate_mlp[-1].bias[1] = 2.0

    def forward(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anomaly_map: [B, H, W]
        Returns:
            scores: [B]
        """
        if anomaly_map.dim() == 2:
            anomaly_map = anomaly_map.unsqueeze(0)
        B = anomaly_map.shape[0]
        flat = anomaly_map.reshape(B, -1)
        num_el = flat.shape[-1]

        pooled_list = []
        for r in self.ratios:
            k = max(1, int(math.ceil(num_el * r)))
            tk = torch.topk(flat, k, dim=-1).values.mean(dim=-1)
            pooled_list.append(tk)
        pooled = torch.stack(pooled_list, dim=-1)  # [B, R]

        # Global map statistics for dynamic ratio selection
        map_mean = flat.mean(dim=-1, keepdim=True)
        map_std = flat.std(dim=-1, keepdim=True)
        stats = torch.cat([pooled.detach(), map_mean.detach(), map_std.detach()], dim=-1)
        weights = F.softmax(self.gate_mlp(stats), dim=-1)  # [B, R]
        return (pooled * weights).sum(dim=-1)


def compute_edge_consistency_loss(pred_prob: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Morphological gradient edge Dice consistency loss (DCP-SFR Eq. 10).
    Args:
        pred_prob: [B, H, W] or [B, 1, H, W] in [0, 1]
        gt_mask: [B, H, W] or [B, 1, H, W] in {0, 1}
    """
    if pred_prob.dim() == 3:
        pred_prob = pred_prob.unsqueeze(1)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    gt_float = gt_mask.float()

    # Morphological gradient via 3x3 max-pool minus min-pool
    pred_dil = F.max_pool2d(pred_prob, kernel_size=3, stride=1, padding=1)
    pred_ero = -F.max_pool2d(-pred_prob, kernel_size=3, stride=1, padding=1)
    pred_edge = (pred_dil - pred_ero).clamp(0.0, 1.0)

    gt_dil = F.max_pool2d(gt_float, kernel_size=3, stride=1, padding=1)
    gt_ero = -F.max_pool2d(-gt_float, kernel_size=3, stride=1, padding=1)
    gt_edge = (gt_dil - gt_ero).clamp(0.0, 1.0)

    B = pred_edge.shape[0]
    p_flat = pred_edge.view(B, -1)
    g_flat = gt_edge.view(B, -1)
    inter = (p_flat * g_flat).sum(dim=1)
    union = (p_flat * p_flat).sum(dim=1) + (g_flat * g_flat).sum(dim=1)
    dice = (2.0 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()


def compute_confidence_weighted_seg_loss(
    similarity_map_list: list,
    gt: torch.Tensor,
    log_conf: torch.Tensor,
    loss_focal,
    loss_dice,
    alpha_conf: float = 0.05,
    beta_edge: float = 0.3,
) -> torch.Tensor:
    """
    Combines AnomalyVFM confidence-weighted pixel loss with DCP-SFR edge consistency loss.
    """
    if gt.dim() == 4:
        gt_sq = gt.squeeze(1)
    else:
        gt_sq = gt

    # Confidence map C in [1, 1 + e] (AnomalyVFM Eq. 4)
    C_map = 1.0 + torch.exp(torch.tanh(log_conf))  # [B, H, W]
    conf_scale = C_map.mean()
    conf_reg = -alpha_conf * torch.log(C_map).mean()

    seg_losses = []
    for sim_map in similarity_map_list:
        f_loss = loss_focal(sim_map, gt)
        d_loss = loss_dice(sim_map[:, 1, :, :], gt)
        e_loss = compute_edge_consistency_loss(sim_map[:, 1, :, :], gt_sq)
        seg_losses.append((f_loss + d_loss) * conf_scale + conf_reg + beta_edge * e_loss)

    return sum(seg_losses)
