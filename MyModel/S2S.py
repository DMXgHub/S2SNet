import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology_propagation.topology_ssm import TopologySSM
from .DINOv3_UNet import DINOv3_UNet
from .MLPDecoder import DecoderHead






def load_text_encoder(text_encoder, device, longclip_repo=None, longclip_checkpoint=None, clip_model="ViT-B/32"):
    text_encoder = (text_encoder or "clip").lower()
    if text_encoder == "clip":
        try:
            import clip
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI CLIP is required when text_encoder='clip'. "
            ) from exc
        model, _ = clip.load(clip_model, device=device)
        return model, lambda texts: clip.tokenize(texts).to(device), "clip"

    if text_encoder == "longclip":
        repo = longclip_repo or os.getenv("LONGCLIP_REPO", "")
        checkpoint = longclip_checkpoint or os.getenv("LONGCLIP_CHECKPOINT", "")
        if not repo:
            raise RuntimeError("Long-CLIP requires longclip_repo in config or LONGCLIP_REPO")
        if not checkpoint:
            raise RuntimeError("Long-CLIP requires longclip_checkpoint in config or LONGCLIP_CHECKPOINT")
        repo_path = os.path.abspath(os.path.expanduser(repo))
        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint))
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"Long-CLIP repo not found: {repo_path}")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Long-CLIP checkpoint not found: {checkpoint_path}")
        if os.path.isdir(os.path.join(repo_path, "model")):
            import_root = os.path.dirname(repo_path)
        elif os.path.basename(repo_path) == "model":
            import_root = os.path.dirname(os.path.dirname(repo_path))
        else:
            import_root = repo_path
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
        from LongCLIP.model import longclip

        model, _ = longclip.load(checkpoint_path, device=device)
        return model, lambda texts: longclip.tokenize(texts).to(device), "longclip"

    raise ValueError(f"unsupported text encoder: {text_encoder}")


class FCP(nn.Module):

    def __init__(self, embed_dim, patch_size=6, memory_size=64, latent_dim=96, temperature=0.1, lambda_factor=0.8):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.memory_size = memory_size
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.lambda_factor = lambda_factor
        self.in_features = embed_dim * patch_size * patch_size

        self.low_freq_filter = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),
            nn.Sigmoid()
        )
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, max(latent_dim * 2, 128)),
            nn.GELU(),
            nn.Linear(max(latent_dim * 2, 128), latent_dim),
            nn.GELU()
        )
        self.memory_bank = nn.Parameter(torch.randn(memory_size, latent_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim * 2, 128)),
            nn.GELU(),
            nn.Linear(max(latent_dim * 2, 128), self.in_features)
        )
        self.out_norm = nn.GroupNorm(8, embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.last_memory_weights = None

    def _center_crop(self, x):
        _, _, height, width = x.shape
        half = self.patch_size // 2
        center_y, center_x = height // 2, width // 2
        top = center_y - half
        left = center_x - half
        return x[:, :, top:top + self.patch_size, left:left + self.patch_size]

    def _center_pad(self, patch, height, width):
        pad_left = (width - self.patch_size) // 2
        pad_right = width - self.patch_size - pad_left
        pad_top = (height - self.patch_size) // 2
        pad_bottom = height - self.patch_size - pad_top
        return F.pad(patch, [pad_left, pad_right, pad_top, pad_bottom], value=1.0).contiguous()

    def forward(self, r_feat, t_feat):
        dtype = r_feat.dtype
        batch_size, channels, height, width = r_feat.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"feature map {height}x{width} is smaller than frequency patch {self.patch_size}x{self.patch_size}"
            )

        r_fft = torch.fft.fft2(r_feat.float(), norm='ortho')
        t_fft = torch.fft.fft2(t_feat.float(), norm='ortho')
        r_fft_centered = torch.fft.fftshift(r_fft, dim=(-2, -1))
        t_fft_centered = torch.fft.fftshift(t_fft, dim=(-2, -1))

        r_amp = torch.abs(r_fft_centered)
        t_amp = torch.abs(t_fft_centered)

        r_patch = self._center_crop(r_amp)
        t_patch = self._center_crop(t_amp)
        filter_gate = self.low_freq_filter(torch.cat([r_patch, t_patch], dim=1))
        filtered_patch = r_patch * (1.0 - filter_gate) + t_patch * filter_gate
        encoder_input = torch.log1p(filtered_patch)

        freq_code = self.encoder(encoder_input)
        freq_code = F.normalize(freq_code, dim=-1)
        memory_norm = F.normalize(self.memory_bank, dim=-1)
        addressing = F.softmax(torch.matmul(freq_code, memory_norm.t()) / self.temperature, dim=-1)
        memory_feature = torch.matmul(addressing, self.memory_bank)
        self.last_memory_weights = addressing.detach()

        decoded_prompt = self.decoder(memory_feature)
        decoded_prompt = torch.sigmoid(decoded_prompt).view(batch_size, channels, self.patch_size, self.patch_size) * 2.0
        decoded_prompt = (1.0 - self.lambda_factor) + self.lambda_factor * decoded_prompt
        full_prompt = self._center_pad(decoded_prompt, height, width)

        prompted_amp = r_amp * full_prompt
        phase_unit = r_fft_centered / r_amp.clamp_min(1e-6)
        prompted_fft_centered = prompted_amp * phase_unit
        prompted_fft = torch.fft.ifftshift(prompted_fft_centered, dim=(-2, -1))
        prompted_feat = torch.fft.ifft2(prompted_fft, dim=(-2, -1), norm='ortho').real.type(dtype)
        return self.out_proj(self.out_norm(prompted_feat))

class DUP(nn.Module):

    def __init__(self, embed_dim, num_hypotheses=16):
        super().__init__()
        self.num_hypotheses = num_hypotheses


        self.evidence_proj = nn.Conv2d(embed_dim, num_hypotheses, 1, bias=False)


        self.expert_feat_proj = nn.Sequential(
            nn.Conv2d(embed_dim, num_hypotheses, 1, bias=False),
            nn.BatchNorm2d(num_hypotheses),
            nn.GELU()
        )

        self.expert_var_proj = nn.Sequential(
            nn.Conv2d(embed_dim, num_hypotheses, 1, bias=False),
            nn.BatchNorm2d(num_hypotheses),
            nn.GELU()
        )


        hidden_dim = max(embed_dim // 4, 32)
        self.energy_head = nn.Sequential(
            nn.Conv2d(embed_dim * 3, hidden_dim, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1)
        )
        self.integration_gate = nn.Sequential(
            nn.Conv2d(embed_dim + 1, hidden_dim, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        self.gate_min = 0.05
        self.gate_max = 0.95

        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(4, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.last_uncertainty_map = None
        self.last_support_gate = None
        self.last_energy = None

    def forward(self, r_feat, t_feat):

        Z_norm = F.normalize(r_feat, p=2, dim=1)

        W = self.evidence_proj.weight
        A_star = W - W.mean(dim=0, keepdim=True)

        W_pos = F.relu(A_star)
        W_neg = F.relu(-A_star)

        Z_pos = F.relu(Z_norm)
        Z_neg = F.relu(-Z_norm)

        e_pos = F.conv2d(Z_pos, W_pos) + F.conv2d(Z_neg, W_neg)
        e_neg = F.conv2d(Z_neg, W_pos) + F.conv2d(Z_pos, W_neg)

        e_pos = torch.clamp(e_pos, min=-10.0, max=10.0)
        e_neg = torch.clamp(e_neg, min=-10.0, max=10.0)

        gate_tokens = torch.cat([r_feat, t_feat, torch.abs(r_feat - t_feat)], dim=1)
        energy = torch.clamp(self.energy_head(gate_tokens), min=-10.0, max=10.0)
        energy_signal = torch.sigmoid(energy)
        support_gate = self.integration_gate(torch.cat([r_feat, energy_signal], dim=1))
        support_gate = self.gate_min + (self.gate_max - self.gate_min) * support_gate
        gated_e_pos = e_pos * support_gate

        exp_e_pos = torch.exp(gated_e_pos)
        exp_neg_e_neg = torch.exp(-e_neg)

        sum_exp_e_pos = exp_e_pos.sum(dim=1, keepdim=True)
        K = self.num_hypotheses

        eta_pos = (exp_e_pos - 1) / torch.clamp(sum_exp_e_pos - K + 1, min=1e-5)
        prod_neg = torch.prod(1 - exp_neg_e_neg + 1e-6, dim=1, keepdim=True)
        eta_neg = 1 - (exp_neg_e_neg / torch.clamp(1 - prod_neg, min=1e-5))
        con = torch.sum(eta_pos * eta_neg, dim=1, keepdim=True)
        gap = torch.sum(exp_neg_e_neg, dim=1, keepdim=True) / K
        con = torch.clamp(con, min=0.0, max=1.0)
        gap = torch.clamp(gap, min=0.0, max=1.0)


        with torch.autocast(device_type=t_feat.device.type, enabled=False):
            w_i = F.softmax(gated_e_pos.float(), dim=1)
            f_i_f32 = self.expert_feat_proj(t_feat).float()
            sigma_sq_f32 = F.softplus(self.expert_var_proj(t_feat).float()) + 1e-6

            expected_sq = torch.sum(w_i * (sigma_sq_f32 + f_i_f32 ** 2), dim=1, keepdim=True)
            sq_expected = torch.sum(w_i * f_i_f32, dim=1, keepdim=True) ** 2
            V = expected_sq - sq_expected
            V = torch.clamp(V, min=0.0, max=1.0)


        euq_features = torch.cat([con, gap, V, 1.0 - support_gate], dim=1)


        uncertainty_map = self.uncertainty_net(euq_features)
        self.last_uncertainty_map = uncertainty_map
        self.last_support_gate = support_gate
        self.last_energy = energy


        out = t_feat * uncertainty_map

        return self.fusion_proj(out)



class GRP(nn.Module):


    def __init__(self, embed_dim, num_nodes=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.scale = (embed_dim // 2) ** -0.5

        self.q_r = nn.Linear(embed_dim, embed_dim // 2)
        self.k_t = nn.Linear(embed_dim, embed_dim // 2)
        self.v_t = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Sequential(
            nn.GroupNorm(8, embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 1)
        )
        self.topology_ssm = TopologySSM(embed_dim, 16, 2.0, d_conv=3, dropout=0.1)

    def forward(self, r_feat, t_feat):
        B, C, H, W = r_feat.shape
        grid_size = int(self.num_nodes ** 0.5)

        r_nodes = F.adaptive_avg_pool2d(r_feat, grid_size).view(B, C, -1).transpose(1, 2)
        t_nodes = F.adaptive_avg_pool2d(t_feat, grid_size).view(B, C, -1).transpose(1, 2)


        Q = self.q_r(r_nodes)
        K = self.k_t(t_nodes)
        V = self.v_t(t_nodes)


        Q_scaled = Q * self.scale
        adj_matrix = F.softmax(torch.bmm(Q_scaled, K.transpose(1, 2)), dim=-1)


        gcn_out = torch.bmm(adj_matrix, V)


        prompt = gcn_out.transpose(1, 2).view(B, C, grid_size, grid_size)
        prompt = self.out_proj(prompt)

        prompt = prompt.permute(0, 2, 3, 1).contiguous()
        prompt = self.topology_ssm(prompt)


        prompt = prompt.permute(0, 3, 1, 2).contiguous()
        return F.interpolate(prompt, size=(H, W), mode='bilinear', align_corners=False)

class FisherStableTextMoE(nn.Module):
    def __init__(self, vis_dim, embed_dim, num_experts=4, fi_temperature=0.5):
        super().__init__()
        self.num_experts = num_experts
        self.fi_temperature = fi_temperature
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(vis_dim, vis_dim // 2),
            nn.GELU(),
            nn.Linear(vis_dim // 2, num_experts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, embed_dim)
            )
            for _ in range(num_experts)
        ])
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.last_routing_weights = None
        self.last_fi_proxy = None

    def forward(self, vis_feat, text_features):
        batch_size = vis_feat.size(0)
        if text_features.dim() == 2:
            base_proto = text_features.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            base_proto = text_features
        expert_residuals = torch.stack([expert(base_proto) for expert in self.experts], dim=1)

        raw_weights = F.softmax(self.router(vis_feat), dim=-1)
        fi_proxy = expert_residuals.float().pow(2).mean(dim=(2, 3))

        if self.training:
            fi_norm = fi_proxy / fi_proxy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            mod_weights = raw_weights * torch.exp(self.fi_temperature * (1.0 - fi_norm.detach()))
            routing_weights = (mod_weights + 1e-4) / (mod_weights.sum(dim=-1, keepdim=True) + 1e-4 * self.num_experts)
        else:
            routing_weights = raw_weights

        residual = torch.einsum('be,bekd->bkd', routing_weights, expert_residuals)
        self.last_routing_weights = routing_weights
        self.last_fi_proxy = fi_proxy.detach()
        return F.normalize(base_proto + self.res_scale * residual, dim=-1)

    def regularization_terms(self):
        device = self.res_scale.device
        if self.last_routing_weights is None or self.last_fi_proxy is None:
            zero = torch.zeros((), device=device)
            return zero, zero

        mean_expert = self.last_routing_weights.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(mean_expert ** 2) - 1.0
        fi_loss = torch.sum(self.last_routing_weights * self.last_fi_proxy, dim=-1).mean()
        return balance_loss, fi_loss


class PrototypeContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.05, min_pixels=1):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.min_pixels = min_pixels
        self.last_logits = None
        self.last_probabilities = None
        self.last_targets = None
        self.last_labels = None
        self.last_sample_ids = None
        self.last_num_classes = None
        self.last_logit_scale = None

    def _clear_cache(self, num_classes=None):
        self.last_logits = None
        self.last_probabilities = None
        self.last_targets = None
        self.last_labels = None
        self.last_sample_ids = None
        self.last_num_classes = num_classes
        self.last_logit_scale = None

    def forward(self, visual_features, text_prototypes, semantic_gt):
        if visual_features.dim() != 4:
            raise ValueError(f"visual_features must be BxCxHxW, got {tuple(visual_features.shape)}")
        if text_prototypes.dim() != 3:
            raise ValueError(f"text_prototypes must be BxKxC, got {tuple(text_prototypes.shape)}")
        if semantic_gt is None:
            return visual_features.new_zeros(())

        batch_size, num_classes, dim = text_prototypes.shape
        self._clear_cache(num_classes=num_classes)

        if visual_features.size(0) != batch_size:
            raise ValueError("visual_features and text_prototypes batch sizes do not match")
        if visual_features.size(1) != dim:
            raise ValueError(
                f"visual dim {visual_features.size(1)} does not match text prototype dim {dim}"
            )

        if semantic_gt.dim() == 4:
            if semantic_gt.size(1) == 1:
                semantic_gt = semantic_gt[:, 0]
            else:
                semantic_gt = semantic_gt.argmax(dim=1)
        if semantic_gt.dim() != 3:
            raise ValueError(f"semantic_gt must be BxHxW or Bx1xHxW, got {tuple(semantic_gt.shape)}")

        target = F.interpolate(
            semantic_gt.unsqueeze(1).float(),
            size=visual_features.shape[-2:],
            mode="nearest"
        ).squeeze(1).long()

        valid_pixels = (target >= 0) & (target < num_classes)
        safe_target = target.clamp(min=0, max=num_classes - 1)
        masks = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        masks = masks * valid_pixels.unsqueeze(1).float()

        counts = masks.sum(dim=(2, 3))
        valid_classes = counts >= float(self.min_pixels)
        if not valid_classes.any():
            return visual_features.new_zeros(())

        pooled_visual = torch.einsum("bkhw,bchw->bkc", masks, visual_features.float())
        pooled_visual = pooled_visual / counts.clamp_min(1.0).unsqueeze(-1)

        class_ids = torch.arange(num_classes, device=visual_features.device).unsqueeze(0).expand(batch_size, -1)
        sample_ids = torch.arange(batch_size, device=visual_features.device).unsqueeze(1).expand(-1, num_classes)
        visual_samples = pooled_visual[valid_classes]
        labels = class_ids[valid_classes]
        sample_ids = sample_ids[valid_classes]

        visual_samples = F.normalize(visual_samples.float(), dim=-1)
        text_prototypes = F.normalize(text_prototypes.float(), dim=-1)
        same_image_text = text_prototypes.index_select(0, sample_ids)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * torch.einsum("nd,nkd->nk", visual_samples, same_image_text)
        targets = F.one_hot(labels, num_classes=num_classes).float()
        probabilities = F.softmax(logits, dim=1)

        self.last_logits = logits.detach()
        self.last_probabilities = probabilities.detach()
        self.last_targets = targets.detach()
        self.last_labels = labels.detach()
        self.last_sample_ids = sample_ids.detach()
        self.last_num_classes = num_classes
        self.last_logit_scale = logit_scale.detach()

        return F.cross_entropy(logits, labels)


class DenseSparseSemanticPrototypeBank(nn.Module):
    def __init__(
            self,
            device,
            vis_dim=1024,
            num_experts=4,
            text_encoder="clip",
            longclip_repo=None,
            longclip_checkpoint=None,
            clip_model="ViT-B/32",
            initialize_text_encoder=True,
            embed_dim=512):
        super().__init__()
        self.device = device
        self.text_encoder = text_encoder
        self.classes = ['unlabeled', 'road', 'sidewalk', 'person', 'car', 'bicycle']
        self.short_prompts = [
            "unlabeled region",
            "road region",
            "sidewalk region",
            "person region",
            "car region",
            "bicycle region",
        ]

        if initialize_text_encoder:
            text_model, tokenizer, encoder_name = load_text_encoder(
                text_encoder=text_encoder,
                device=device,
                longclip_repo=longclip_repo,
                longclip_checkpoint=longclip_checkpoint,
                clip_model=clip_model
            )
            text_model.eval()
            for param in text_model.parameters():
                param.requires_grad = False
            self.text_encoder_name = encoder_name
            short_features = self._encode_texts(text_model, tokenizer, self.short_prompts)
            fallback_dense_feature = self._encode_texts(text_model, tokenizer, ["an RGB thermal street scene"])
            self.embed_dim = short_features.size(-1)
            del text_model
            if torch.cuda.is_available() and torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        else:
            self.text_encoder_name = (text_encoder or "longclip").lower()
            self.embed_dim = int(embed_dim)
            short_features = torch.zeros(len(self.short_prompts), self.embed_dim, dtype=torch.float32)
            fallback_dense_feature = torch.zeros(1, self.embed_dim, dtype=torch.float32)

        embed_dim = self.embed_dim
        self.short_moe = FisherStableTextMoE(vis_dim, embed_dim, num_experts=num_experts)
        self.long_moe = FisherStableTextMoE(vis_dim, embed_dim, num_experts=num_experts)
        self.dense_conditioner = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.visual_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(vis_dim, embed_dim),
            nn.GELU()
        )
        self.stream_fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2)
        )
        self.register_buffer("short_text_features", short_features)
        self.register_buffer("fallback_dense_feature", fallback_dense_feature)
        self.last_stream_weights = None
        self.last_dense_text_features = None

    def _encode_texts(self, text_model, tokenizer, prompts):
        tokens = tokenizer(prompts)
        with torch.no_grad():
            features = text_model.encode_text(tokens).float()
            features = F.normalize(features, dim=-1)
        return features

    def forward(self, vis_feat, dense_text_features=None):
        batch_size = vis_feat.size(0)
        short_proto = self.short_moe(vis_feat, self.short_text_features)

        if dense_text_features is None:
            dense_text_features = self.fallback_dense_feature.expand(batch_size, -1)
        elif dense_text_features.dim() == 1:
            dense_text_features = dense_text_features.unsqueeze(0)
        if dense_text_features.size(-1) != self.embed_dim:
            raise ValueError(
                f"dense_text_features has dim {dense_text_features.size(-1)}, "
                f"but {self.text_encoder_name} expects {self.embed_dim}. "
                "Rebuild dense text features with the same text encoder."
            )
        fallback_dense = self.fallback_dense_feature.expand(batch_size, -1)
        zero_mask = dense_text_features.norm(dim=-1, keepdim=True) < 1e-6
        dense_text_features = torch.where(zero_mask, fallback_dense, dense_text_features)
        dense_text_features = F.normalize(dense_text_features.float(), dim=-1)
        self.last_dense_text_features = dense_text_features

        short_base = self.short_text_features.unsqueeze(0).expand(batch_size, -1, -1)
        dense_context = dense_text_features.unsqueeze(1).expand(-1, len(self.classes), -1)
        dense_proto_input = self.dense_conditioner(torch.cat([short_base, dense_context], dim=-1))
        dense_proto_input = F.normalize(dense_proto_input, dim=-1)
        long_proto = self.long_moe(vis_feat, dense_proto_input)

        fusion_tokens = torch.cat([
            short_proto,
            long_proto,
            torch.abs(short_proto - long_proto),
            short_proto * long_proto,
        ], dim=-1)
        stream_weights = F.softmax(self.stream_fusion(fusion_tokens), dim=-1)
        class_proto = stream_weights[..., :1] * short_proto + stream_weights[..., 1:] * long_proto
        self.last_stream_weights = stream_weights
        return F.normalize(class_proto, dim=-1)

    def regularization_terms(self):
        short_balance, short_fi = self.short_moe.regularization_terms()
        long_balance, long_fi = self.long_moe.regularization_terms()

        if self.last_stream_weights is None:
            stream_balance = short_balance.new_tensor(0.0)
        else:
            mean_stream = self.last_stream_weights.mean(dim=(0, 1))
            stream_balance = 2.0 * torch.sum(mean_stream ** 2) - 1.0

        return {
            "semantic_balance": short_balance + long_balance,
            "semantic_fi": short_fi + long_fi,
            "stream_balance": stream_balance,
        }

class StableResidualFusion(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.alpha_p = nn.Parameter(torch.ones(1) * 0.01)
        self.alpha_t = nn.Parameter(torch.ones(1) * 0.01)
        hidden_dim = max(embed_dim // 4, 16)
        self.prompt_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU()
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim * 4, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim * 2, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(embed_dim * 4, hidden_dim, 1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),
            nn.Sigmoid()
        )
        self.norm = nn.GroupNorm(8, embed_dim)

    def forward(self, r_feat, t_feat, prompt_feat):

        prompt_feat = self.prompt_refine(prompt_feat)
        quality_tokens = torch.cat([
            r_feat,
            t_feat,
            prompt_feat,
            torch.abs(r_feat - t_feat)
        ], dim=1)
        channel_t, channel_p = self.channel_gate(quality_tokens).chunk(2, dim=1)
        spatial_t, spatial_p = self.spatial_gate(quality_tokens).chunk(2, dim=1)
        fused = r_feat + t_feat * channel_t * spatial_t * self.alpha_t + prompt_feat * channel_p * spatial_p * self.alpha_p
        return self.norm(fused)


class SemanticAwareTopFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = max(embed_dim // 4, 64)
        self.rgb_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU()
        )
        self.thermal_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU()
        )
        self.modality_gate = nn.Sequential(
            nn.Conv2d(embed_dim * 4, hidden_dim, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim * 2, 1),
            nn.Sigmoid()
        )
        self.res_scale = nn.Parameter(torch.ones(1) * 0.2)
        self.norm = nn.GroupNorm(8, embed_dim)

    def forward(self, r_feat, t_feat):
        r_ref = self.rgb_refine(r_feat)
        t_ref = self.thermal_refine(t_feat)
        fusion_tokens = torch.cat([
            r_feat,
            t_feat,
            torch.abs(r_feat - t_feat),
            r_feat + t_feat
        ], dim=1)
        gate_r, gate_t = self.modality_gate(fusion_tokens).chunk(2, dim=1)
        fused = r_feat + t_feat + self.res_scale * (gate_r * r_ref + gate_t * t_ref)
        return self.norm(fused)


class PoincareMath:


    @staticmethod
    def expmap0(u, c, eps=1e-4):

        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, keepdim=True), eps)
        gamma_1 = torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)

        max_norm = (1-eps)/sqrt_c
        gamma_1_norm = torch.clamp_min(gamma_1.norm(dim=-1, keepdim=True), 1e-7)
        cond = gamma_1_norm > max_norm
        projected = torch.where(cond, gamma_1 / gamma_1_norm * max_norm, gamma_1)
        return projected

    @staticmethod
    def poincare_distance(x, y, c, eps=1e-5):

        sqrt_c = c ** 0.5


        x_sqnorm = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_sqnorm = torch.sum(y ** 2, dim=-1).unsqueeze(1)
        xy_inner = torch.matmul(x, y.transpose(-1, -2))
        sqdist = x_sqnorm + y_sqnorm - 2 * xy_inner
        sqdist = torch.clamp_min(sqdist, eps)


        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y ** 2, dim=-1).unsqueeze(1)
        denom_x = torch.clamp_min(1 - c * x_norm_sq, eps)
        denom_y = torch.clamp_min(1 - c * y_norm_sq, eps)


        delta = 2 * c * sqdist / (denom_x * denom_y)


        inner = torch.clamp_min(1 + delta, 1 + eps)
        dist = torch.acosh(inner) / sqrt_c
        return dist


class HyperbolicCrossAttention(nn.Module):

    def __init__(self, embed_dim, c=1.0):
        super().__init__()
        self.embed_dim = embed_dim

        self.register_buffer("c", torch.tensor([c], dtype=torch.float32))


        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)


        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):


        with torch.autocast(device_type=query.device.type, enabled=False):
            query = query.float()
            key = key.float()
            value = value.float()

            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            c = F.softplus(self.c.float()).to(query.device) + 1e-4
            q_hyp = PoincareMath.expmap0(q, c)
            k_hyp = PoincareMath.expmap0(k, c)

            dist_matrix = PoincareMath.poincare_distance(q_hyp, k_hyp, c)
            attn_logits = - F.softplus(self.alpha.float()) * dist_matrix
            attn_weights = F.softmax(attn_logits, dim=-1)

            out = torch.matmul(attn_weights, v)

        out = self.out_proj(out)
        return out


class PromptRouter(nn.Module):
    def __init__(self, embed_dim, num_prompts=3, hidden_dim=None, init_temperature=1.0, hard=True):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 4, 32)
        self.hard = hard
        self.temperature = nn.Parameter(torch.tensor(float(init_temperature)))
        self.router = nn.Sequential(
            nn.Conv2d(embed_dim * 3, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_prompts)
        )
        self.last_logits = None
        self.last_weights = None
        self.last_balance_probs = None
        self.last_choice = None

    def forward(self, r_feat, t_feat):
        route_input = torch.cat([r_feat, t_feat, torch.abs(r_feat - t_feat)], dim=1)
        logits = self.router(route_input)
        temperature = self.temperature.clamp(min=0.2, max=5.0)
        probs = F.softmax(logits / temperature, dim=-1)

        if self.hard:
            index = probs.argmax(dim=-1)
            hard_weights = F.one_hot(index, num_classes=probs.size(-1)).to(dtype=probs.dtype)
            weights = hard_weights.detach() - probs.detach() + probs
        else:
            index = probs.argmax(dim=-1)
            weights = probs

        self.last_logits = logits.detach()
        self.last_weights = probs.detach()
        self.last_balance_probs = probs
        self.last_choice = index.detach()
        return weights

class RoutedPromptBank(nn.Module):
    def __init__(self, embed_dim, num_nodes=64, hard_router=True):
        super().__init__()
        self.prompt_names = ["frequency", "uncertainty", "topology"]
        self.frequency_prompt = FCP(embed_dim)
        self.uncertainty_prompt = DUP(embed_dim)
        self.topology_prompt = GRP(embed_dim, num_nodes=num_nodes)
        self.router = PromptRouter(embed_dim, num_prompts=3, hard=hard_router)

        self.last_prompt_weights = None
        self.last_prompt_probs = None
        self.last_prompt_choice = None

    def forward(self, r_feat, t_feat):
        p_freq = self.frequency_prompt(r_feat, t_feat)
        p_uncertainty = self.uncertainty_prompt(r_feat, t_feat)
        p_topology = self.topology_prompt(r_feat, t_feat)

        candidates = torch.stack([p_freq, p_uncertainty, p_topology], dim=1)
        weights = self.router(r_feat, t_feat)
        mixed_prompt = (candidates * weights[:, :, None, None, None]).sum(dim=1)

        self.last_prompt_weights = self.router.last_weights
        self.last_prompt_probs = self.router.last_balance_probs
        self.last_prompt_choice = self.router.last_choice
        return mixed_prompt


class S2S(nn.Module):
    def __init__(
            self,
            num_classes=6,
            text_encoder="clip",
            longclip_repo=None,
            longclip_checkpoint=None,
            clip_model="ViT-B/32",
            beta_cal_beta=0.2,
            beta_cal_weight=0.005,
            proto_contrast_temperature=0.05,
            proto_contrast_weight=None,
            prototype_kl_weight=1e-3,
            prompt_balance_weight=0.01,
            prompt_confidence_weight=0.001,
            prompt_route_history_maxlen=None,
            dinov3_weight_path=None,
            dinov3_local_path=None,
            freeze_backbone=True,
            initialize_text_encoder=True,
            text_embed_dim=512,
            device=None):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.prototype_kl_weight = prototype_kl_weight
        self.prompt_balance_weight = prompt_balance_weight
        self.prompt_confidence_weight = prompt_confidence_weight
        self.prompt_route_history_maxlen = prompt_route_history_maxlen
        self.prompt_route_history = []
        self.last_prompt_route_record = None
        self.prompt_route_step = 0
        self.backbone = DINOv3_UNet(
            dinov3_weight_path=dinov3_weight_path,
            dinov3_local_path=dinov3_local_path,
            freeze_backbone=freeze_backbone,
        )
        self.channels_list = [96, 192, 384, 768]


        self.hy_prompt_1 = RoutedPromptBank(self.channels_list[0],num_nodes=64)
        self.hy_prompt_2 = RoutedPromptBank(self.channels_list[1],64)
        self.hy_prompt_3 = RoutedPromptBank(self.channels_list[2],64)
        self.hy_prompt_4 = RoutedPromptBank(self.channels_list[3], 64)

        self.fusion_s1 = StableResidualFusion(self.channels_list[0])
        self.fusion_s2 = StableResidualFusion(self.channels_list[1])
        self.fusion_s3 = StableResidualFusion(self.channels_list[2])
        self.fusion_s4 = StableResidualFusion(self.channels_list[3])
        self.top_fusion = SemanticAwareTopFusion(self.channels_list[3])
        self.semantic_bank = DenseSparseSemanticPrototypeBank(
            self.device,
            vis_dim=self.channels_list[3],
            text_encoder=text_encoder,
            longclip_repo=longclip_repo,
            longclip_checkpoint=longclip_checkpoint,
            clip_model=clip_model,
            initialize_text_encoder=initialize_text_encoder,
            embed_dim=text_embed_dim,
        )
        self.text_proj = nn.Linear(self.semantic_bank.embed_dim, self.channels_list[3])

        self.hyper_attn = HyperbolicCrossAttention(self.channels_list[3], c=1.0)

        self.proto_contrast = PrototypeContrastiveLoss(temperature=proto_contrast_temperature)
        self.proto_contrast_weight = beta_cal_weight if proto_contrast_weight is None else proto_contrast_weight
        self.last_proto_contrast_loss = None
        self.last_proto_visual_features = None
        self.last_proto_text_prototypes = None

        self.prototype_temperature = nn.Parameter(torch.ones(1) * 10.0)
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(self.channels_list[3] * 2, self.channels_list[3] // 4, 1, bias=False),
            nn.GroupNorm(8, self.channels_list[3] // 4),
            nn.GELU(),
            nn.Conv2d(self.channels_list[3] // 4, 1, 1),
            nn.Sigmoid()
        )
        self.semantic_refine = nn.Sequential(
            nn.Conv2d(self.channels_list[3], self.channels_list[3], 3, padding=1, groups=self.channels_list[3], bias=False),
            nn.Conv2d(self.channels_list[3], self.channels_list[3], 1, bias=False),
            nn.GroupNorm(8, self.channels_list[3])
        )
        self.gamma = nn.Parameter(torch.ones(1) * 0.05)

        self.decoder = DecoderHead(self.channels_list, num_classes)

    def reset_prompt_route_history(self):
        self.prompt_route_history.clear()
        self.last_prompt_route_record = None
        self.prompt_route_step = 0

    def get_prompt_route_history(self):
        return list(self.prompt_route_history)

    def _format_prompt_route_choice(self, prompt_bank):
        choice = prompt_bank.last_prompt_choice
        weights = prompt_bank.last_prompt_weights
        if choice is None:
            return None

        choice_ids = choice.detach().cpu().tolist()
        if not isinstance(choice_ids, list):
            choice_ids = [choice_ids]
        choice_ids = [int(choice_id) for choice_id in choice_ids]
        choice_names = [prompt_bank.prompt_names[choice_id] for choice_id in choice_ids]

        counts = {name: 0 for name in prompt_bank.prompt_names}
        for choice_name in choice_names:
            counts[choice_name] += 1

        record = {
            "choice_ids": choice_ids,
            "choice_names": choice_names,
            "counts": counts,
        }
        if weights is not None:
            record["weights"] = weights.detach().cpu().tolist()
        return record

    def _record_prompt_routes(self):
        stage_banks = [
            ("stage1", self.hy_prompt_1),
            ("stage2", self.hy_prompt_2),
            ("stage3", self.hy_prompt_3),
            ("stage4", self.hy_prompt_4),
        ]
        stages = {
            stage_name: self._format_prompt_route_choice(prompt_bank)
            for stage_name, prompt_bank in stage_banks
        }
        record = {
            "step": self.prompt_route_step,
            "training": bool(self.training),
            "stages": stages,
        }
        self.last_prompt_route_record = record
        self.prompt_route_history.append(record)
        self.prompt_route_step += 1

        if self.prompt_route_history_maxlen is not None:
            overflow = len(self.prompt_route_history) - int(self.prompt_route_history_maxlen)
            if overflow > 0:
                del self.prompt_route_history[:overflow]

    def _get_prompt_balance_terms(self):
        prompt_banks = [self.hy_prompt_1, self.hy_prompt_2, self.hy_prompt_3]
        balance_losses = []
        confidence_losses = []
        hard_load_losses = []

        for prompt_bank in prompt_banks:
            probs = prompt_bank.last_prompt_probs
            if probs is None:
                continue

            probs_float = probs.float()
            safe_probs = probs_float.clamp_min(1e-6)
            num_prompts = probs_float.size(-1)
            log_num_prompts = math.log(num_prompts)

            mean_prob = probs_float.mean(dim=0)
            safe_mean_prob = mean_prob.clamp_min(1e-6)
            balance_losses.append(
                torch.sum(safe_mean_prob * (safe_mean_prob.log() + log_num_prompts))
            )

            confidence_losses.append(
                -torch.sum(probs_float * safe_probs.log(), dim=-1).mean() / log_num_prompts
            )

            choice = prompt_bank.last_prompt_choice
            if choice is not None:
                hard_load = F.one_hot(choice, num_classes=num_prompts).float().mean(dim=0)
                hard_load_losses.append(num_prompts * torch.sum(hard_load ** 2) - 1.0)

        if not balance_losses:
            zero = next(self.parameters()).new_zeros(())
            return zero, zero, zero

        balance_loss = torch.stack(balance_losses).mean()
        confidence_loss = torch.stack(confidence_losses).mean()
        if hard_load_losses:
            hard_load_loss = torch.stack(hard_load_losses).mean()
        else:
            hard_load_loss = balance_loss.detach().new_zeros(())
        return balance_loss, confidence_loss, hard_load_loss

    def forward(self, rgb, thermal, dense_text_features=None):
        r1, r2, r3, r4 = self.backbone(rgb)
        t1, t2, t3, t4 = self.backbone(thermal)

        p1 = self.hy_prompt_1(r1,t1)
        f1 = self.fusion_s1(r1,t1,p1)

        p2 = self.hy_prompt_2(r2,t2)
        f2 = self.fusion_s2(r2,t2,p2)

        p3 = self.hy_prompt_3(r3,t3)
        f3 = self.fusion_s3(r3,t3,p3)

        p4 = self.hy_prompt_4(r4,t4)
        f4 = self.fusion_s4(r4,t4,p4)
        self._record_prompt_routes()

        f5 = self.top_fusion(r4, t4)
        b, c, h, w = f5.shape
        f5_base = f5

        class_proto = self.semantic_bank(f5_base, dense_text_features=dense_text_features).float()
        text_p = self.text_proj(class_proto)

        proto_logits = torch.einsum(
            'bchw,bkc->bkhw',
            F.normalize(f5_base.float(), dim=1),
            F.normalize(text_p.float(), dim=-1)
        ) * self.prototype_temperature.clamp(min=1.0, max=30.0)

        proto_weights_float = F.softmax(proto_logits, dim=1)
        proto_weights = proto_weights_float.type_as(f5_base)

        proto_confidence = proto_weights_float.max(dim=1, keepdim=True).values
        proto_entropy = -(proto_weights_float * torch.log(proto_weights_float.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        proto_entropy = proto_entropy / math.log(proto_weights_float.size(1))
        proto_relevance = (proto_confidence * (1.0 - proto_entropy)).clamp(0.0, 1.0).type_as(f5_base)

        self.last_proto_visual_features = f5_base
        self.last_proto_text_prototypes = text_p
        self.last_proto_contrast_loss = None

        proto_context = torch.einsum('bkhw,bkc->bchw', proto_weights, text_p.type_as(f5_base))

        vis_flat = f5_base.flatten(2).transpose(1, 2)
        hyper_context = self.hyper_attn(query=vis_flat, key=text_p, value=text_p)
        hyper_context = hyper_context.transpose(1, 2).reshape(b, c, h, w).type_as(f5_base)

        f5 = (hyper_context+proto_context) * proto_relevance

        outputs = self.decoder([f1,f2,f3,f4,f5])
        outputs = F.interpolate(outputs, size=rgb.shape[2:], mode='bilinear', align_corners=False)

        return outputs

    def get_aux_loss(self, targets=None, seg_logits=None):
        device = next(self.parameters()).device
        aux_loss = torch.zeros((), device=device)
        aux_items = {}

        prompt_balance_loss, prompt_confidence_loss, prompt_hard_load_loss = self._get_prompt_balance_terms()
        prompt_balance_aux = (
            self.prompt_balance_weight * prompt_balance_loss
            + self.prompt_confidence_weight * prompt_confidence_loss
        )
        aux_loss = aux_loss + prompt_balance_aux
        aux_items["prompt_balance"] = prompt_balance_loss.detach()
        aux_items["prompt_confidence"] = prompt_confidence_loss.detach()
        aux_items["prompt_hard_load"] = prompt_hard_load_loss.detach()

        proto_visual = getattr(self, "last_proto_visual_features", None)
        proto_text = getattr(self, "last_proto_text_prototypes", None)
        if targets is not None and proto_visual is not None and proto_text is not None:
            proto_contrast_loss = self.proto_contrast(
                visual_features=proto_visual,
                text_prototypes=proto_text,
                semantic_gt=(targets[0] if isinstance(targets, (tuple, list)) else targets)
            )
            self.last_proto_contrast_loss = proto_contrast_loss
            aux_loss = aux_loss + self.proto_contrast_weight * proto_contrast_loss
            aux_items["ssr_proto"] = proto_contrast_loss.detach()

        return aux_loss, aux_items


if __name__ == "__main__":
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 480, 640)
    thermal = torch.randn(batch_size, 3, 480, 640)
    rgb = rgb.cuda()
    thermal = thermal.cuda()
    model = S2S().cuda()

    output = model(rgb, thermal)
    print(output.shape)