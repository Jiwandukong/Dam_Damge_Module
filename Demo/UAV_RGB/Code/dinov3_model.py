"""The existing fine-tuned DINOv3 ViT-B/16 + UPerNet + RGB refinement network.
Only the complete fine-tuned checkpoint is needed; no training labels are loaded.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

CLASSES = ['CRC', 'DLM', 'SPL', 'LEAK_EFF']
BACKBONE_CONFIG = {'architectures': ['DINOv3ViTModel'], 'attention_dropout': 0.0, 'drop_path_rate': 0.0, 'hidden_act': 'gelu', 'hidden_size': 768, 'image_size': 224, 'initializer_range': 0.02, 'intermediate_size': 3072, 'key_bias': False, 'layer_norm_eps': 1e-05, 'layerscale_value': 1.0, 'mlp_bias': True, 'model_type': 'dinov3_vit', 'num_attention_heads': 12, 'num_channels': 3, 'num_hidden_layers': 12, 'num_register_tokens': 4, 'patch_size': 16, 'pos_embed_jitter': None, 'pos_embed_rescale': 2.0, 'pos_embed_shift': None, 'proj_bias': True, 'query_bias': True, 'rope_theta': 100.0, 'torch_dtype': 'float32', 'transformers_version': '4.56.0.dev0', 'use_gated_mlp': False, 'value_bias': True}

class ConvNormAct(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int, kernel: int = 3):
        # At least two channels per group, including pooled 1x1 tensors.
        groups = math.gcd(32, max(1, outgoing // 2))
        while outgoing % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(incoming, outgoing, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(groups, outgoing),
            nn.ReLU(inplace=True),
        )

class UPerNet(nn.Module):
    def __init__(self, incoming: int = 768, channels: int = 128,
                 pool_scales=(1, 2, 3, 6), dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.adapters = nn.ModuleList([ConvNormAct(incoming, channels, 1) for _ in range(4)])
        self.psp = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(scale), ConvNormAct(channels, channels, 1))
            for scale in pool_scales
        ])
        self.psp_bottleneck = ConvNormAct(channels * (len(pool_scales) + 1), channels)
        self.laterals = nn.ModuleList([ConvNormAct(channels, channels, 1) for _ in range(3)])
        self.fpn = nn.ModuleList([ConvNormAct(channels, channels) for _ in range(3)])
        self.fpn_bottleneck = ConvNormAct(channels * 4, channels)
        self.classifier = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(channels, 1, 1))

    def forward(self, features: tuple, output_size: tuple[int, int]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("UPerNet needs exactly four intermediate feature grids")
        height, width = output_size
        pyramid = []
        for feature, adapter, stride in zip(features, self.adapters, (4, 8, 16, 32)):
            projected = adapter(feature)
            pyramid.append(F.interpolate(projected, size=(height // stride, width // stride),
                                         mode="bilinear", align_corners=False))
        deep = pyramid[-1]
        psp = [deep] + [F.interpolate(branch(deep), size=deep.shape[-2:],
                                     mode="bilinear", align_corners=False) for branch in self.psp]
        laterals = [layer(grid) for layer, grid in zip(self.laterals, pyramid[:-1])]
        laterals.append(self.psp_bottleneck(torch.cat(psp, dim=1)))
        for index in range(3, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index], size=laterals[index - 1].shape[-2:],
                mode="bilinear", align_corners=False)
        fpn = [layer(grid) for layer, grid in zip(self.fpn, laterals[:-1])] + [laterals[-1]]
        fpn = [F.interpolate(grid, size=fpn[0].shape[-2:], mode="bilinear",
                             align_corners=False) for grid in fpn]
        logits = self.classifier(self.fpn_bottleneck(torch.cat(fpn, dim=1)))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

class HFDINOv3BackboneAdapter(nn.Module):
    """Expose the official Transformers DINOv3 through the common grid API.

    v4.56.2 hidden_states includes the input embeddings at index 0. Its output
    capture decorator replaces the last collected block output with the final
    normalized last_hidden_state. Hence block i uses hidden_states[i+1], with
    one model.norm for intermediate blocks and no extra norm for the final
    block. CLS and register tokens are omitted. Production loading below is
    restricted to the official pretrained B/16; small instances in tests are
    random unit tests only, never a production fallback.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.embed_dim = model.config.hidden_size
        self.patch_size = model.config.patch_size
        self.n_blocks = model.config.num_hidden_layers
        self.n_storage_tokens = model.config.num_register_tokens

    def get_intermediate_layers(self, image: torch.Tensor, *, n, reshape: bool = True,
                                norm: bool = True):
        if not norm:
            raise ValueError("HF common comparison supports normalized intermediate features only")
        indices = tuple(range(self.n_blocks - n, self.n_blocks)) if isinstance(n, int) else tuple(n)
        if not indices or min(indices) < 0 or max(indices) >= self.n_blocks:
            raise ValueError("Requested intermediate block is outside the HF backbone")
        outputs = self.model(pixel_values=image, output_hidden_states=True, return_dict=True)
        states = outputs.hidden_states
        if states is None or len(states) != self.n_blocks + 1:
            raise ValueError("Transformers hidden-state indexing changed; refuse silently differing features")
        batch, _, height, width = image.shape
        features = []
        for index in indices:
            state = states[index + 1]
            if index != self.n_blocks - 1:
                state = self.model.norm(state)
            state = state[:, self.n_storage_tokens + 1:]
            if state.shape[1] != (height // self.patch_size) * (width // self.patch_size):
                raise ValueError("HF patch-token count does not match the original feature grid")
            if reshape:
                state = state.reshape(batch, height // self.patch_size,
                                      width // self.patch_size, self.embed_dim).permute(0, 3, 1, 2).contiguous()
            features.append(state)
        return tuple(features)

class DamageSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import DINOv3ViTConfig, AutoModel
        self.backbone = HFDINOv3BackboneAdapter(AutoModel.from_config(
            DINOv3ViTConfig(**BACKBONE_CONFIG), attn_implementation='sdpa'))
        self.decoder = UPerNet(channels=128, dropout=0.0)
        self.decoder.classifier = nn.Conv2d(128, 4, 1)
        self.detail = nn.Sequential(ConvNormAct(7, 32), ConvNormAct(32, 32), nn.Conv2d(32, 4, 1))

    def forward(self, image):
        grids = self.backbone.get_intermediate_layers(image, n=(2, 5, 8, 11), reshape=True, norm=True)
        logits = self.decoder(grids, tuple(image.shape[-2:]))
        return logits + self.detail(torch.cat([image, logits], dim=1))


def load_model(checkpoint_path, device='cuda:0'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = DamageSegmentor()
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint['inference']
