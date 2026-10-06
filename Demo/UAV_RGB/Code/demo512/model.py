"""SAM3 image/text segmentation on an actual 512 by 512 RGB tensor.

The supplied checkpoint is loaded strictly at its original parameter/buffer shapes
before adapting *runtime geometry*. The original 14-pixel patch convolution uses
three pixels of internal zero padding on each side: 512 -> 37 tokens, with no
image resize and no larger RGB canvas. Its receptive field covers all 512 pixels.
The 148-pixel semantic logits represent a 518-pixel padded field, so output
postprocessing interpolates logits to 518 and removes the three-pixel border.

This demo uses SAM3's existing text-conditioned ``semantic_seg`` head. That is a
different readout from the legacy processor's thresholded union of instance
masks. No new trainable head, ground-truth input, or inference-only decorator is
introduced. Training and inference call the same differentiable functions.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BASE_SHA256 = "a2749dba62207575afac9ed42f923d7cdfd7e2d2f0ffb2210a103e91657d985d"
IMAGE_SIZE = 512
PATCH_SIZE = 14
PATCH_PADDING = 3
TOKEN_GRID = 37
PADDED_FIELD_SIZE = TOKEN_GRID * PATCH_SIZE
SEMANTIC_GRID = TOKEN_GRID * 4
CLASS_PROMPTS = {"CRC": "crack", "DLM": "delamination", "SPL": "spalling"}


def _differentiable_vit_mlp(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # The local upstream fused addmm activation explicitly rejects enabled
    # autograd and detaches weights. Execute the exact existing layers instead;
    # both training and inference use this same arithmetic and parameter objects.
    x = mlp.act(mlp.fc1(x))
    x = mlp.drop1(x)
    x = mlp.norm(x)
    x = mlp.fc2(x)
    return mlp.drop2(x)


def _import_upstream():
    """Use installed SAM3, or an explicitly selected development checkout.

    Never discover SAM3 by walking out of the standalone demo folder.
    """
    override = os.environ.get("SAM3_SOURCE")
    selected = Path(override).expanduser().resolve() if override else None
    if override and not (selected / "sam3/model_builder.py").is_file():
        raise ValueError("SAM3_SOURCE must point to the SAM3 repository root")
    explicit_source = selected is not None
    if explicit_source and str(selected) not in sys.path:
        sys.path.insert(0, str(selected))
    import sam3
    from sam3 import model_builder

    actual = Path(model_builder.__file__).resolve()
    if explicit_source and not actual.is_relative_to(selected.resolve()):
        raise RuntimeError(f"SAM3_SOURCE/checkout conflict: imported {actual}")
    if getattr(sam3, "__version__", None) != "0.1.0":
        raise RuntimeError("This adapter requires SAM3 0.1.0; source provenance is recorded")
    return model_builder


def checkpoint_sha256(path: str | Path) -> str:
    """Stream the base checkpoint without loading its optimizer into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_base_state(path: Path) -> dict[str, torch.Tensor]:
    # This training checkpoint contains NumPy RNG arrays alongside model tensors.
    # Permit only the NumPy reconstruction types it needs; never use pickle's
    # unrestricted weights_only=False fallback.
    from numpy.core.multiarray import _reconstruct, scalar

    allowed = [
        _reconstruct,
        scalar,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("The base checkpoint has no model state dictionary")
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("The model state contains a non-tensor entry")
    return state


def _build_original_architecture() -> nn.Module:
    """Original image architecture; omit CUDA-only positional-cache prewarming.

    All learned modules and their settings match build_sam3_image_model. The
    explicit decoder construction changes only resolution=None so CPU builds do
    not allocate an unrelated CUDA coordinate cache. Strict loading below checks
    every learned parameter and persistent buffer against the base checkpoint.
    """
    builder = _import_upstream()
    vision = builder._create_vit_neck(
        builder._create_position_encoding(),
        builder._create_vit_backbone(),
        enable_inst_interactivity=False,
    )
    package_root = Path(builder.__file__).resolve().parent
    text = builder._create_text_encoder(
        str(package_root / "assets/bpe_simple_vocab_16e6.txt.gz")
    )
    backbone = builder._create_vl_backbone(vision, text)
    layer = builder.TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=builder.MultiheadAttention(
            num_heads=8, dropout=0.1, embed_dim=256, use_fa3=False
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )
    decoder = builder.TransformerDecoder(
        layer=layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=None,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    transformer = builder.TransformerWrapper(
        encoder=builder._create_transformer_encoder(),
        decoder=decoder,
        d_model=256,
    )
    return builder._create_sam3_model(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=builder._create_geometry_encoder(),
        segmentation_head=builder._create_segmentation_head(),
        dot_prod_scoring=builder._create_dot_product_scoring(),
        inst_interactive_predictor=None,
        eval_mode=True,
    )


def _configure_native_grid(model: nn.Module) -> None:
    """Adapt only this model instance; no global changes or upstream file edits."""
    shapes_before = {name: tuple(p.shape) for name, p in model.named_parameters()}
    trunk = model.backbone.vision_backbone.trunk
    projection = trunk.patch_embed.proj
    if projection.kernel_size != (PATCH_SIZE, PATCH_SIZE) or projection.stride != (
        PATCH_SIZE,
        PATCH_SIZE,
    ):
        raise RuntimeError("Native-512 adapter requires the original 14px SAM3 patches")
    projection.padding = (PATCH_PADDING, PATCH_PADDING)
    for block in trunk.blocks:
        block.mlp.forward = MethodType(_differentiable_vit_mlp, block.mlp)
        # Local windows keep the original 24x24 RoPE. Only global blocks attend
        # over the entire 37x37 grid. SAM3 already pads/unpads local token windows.
        if block.window_size == 0:
            attention = block.attn
            if attention.use_rel_pos or attention.use_ve_rope:
                raise RuntimeError("Unexpected positional architecture in SAM3 base")
            attention.input_size = (TOKEN_GRID, TOKEN_GRID)
            attention._setup_rope_freqs()
    decoder = model.transformer.decoder
    decoder.compilable_cord_cache = None
    decoder.compilable_stored_size = None
    decoder.coord_cache.clear()
    model.backbone.vision_backbone.position_encoding.cache.clear()
    shapes_after = {name: tuple(p.shape) for name, p in model.named_parameters()}
    if shapes_before != shapes_after:
        raise RuntimeError("Native geometry adaptation changed a learned parameter")


def build_demo_model(
    checkpoint_path: str | Path,
    device: str | torch.device,
    training: bool = False,
    *,
    expected_sha256: str = BASE_SHA256,
    checkpoint_kind: str = "base",
) -> nn.Module:
    """Load a verified base or a complete native-512 demo checkpoint strictly.

    ``training`` controls module train/eval behavior, not requires_grad. The
    caller selects trainable parameter groups. ``forward_class_outputs`` omits
    only the target-matching/loss bookkeeping, so model.train() is supported and
    no ground truth or dummy matcher is needed. Inference calls model.eval().

    For a self-contained fine-tuned checkpoint, pass checkpoint_kind="demo" and
    its recorded expected_sha256. Its 37x37 persistent RoPE buffers are loaded
    after configuring the native grid. No copy of the 10 GB base is needed for
    that inference path. Base loading always checks the immutable known hash
    and strictly loads the original 72x72 buffers before runtime adaptation.
    """
    if checkpoint_kind not in {"base", "demo"}:
        raise ValueError("checkpoint_kind must be 'base' or 'demo'")
    if checkpoint_kind == "base" and expected_sha256 != BASE_SHA256:
        raise ValueError("Base loading requires the immutable base SHA-256")
    if checkpoint_kind == "demo" and expected_sha256 == BASE_SHA256:
        raise ValueError("Demo loading requires the demo checkpoint's recorded SHA-256")
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase 64-digit SHA-256 digest")
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    actual_hash = checkpoint_sha256(path)
    if actual_hash != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )
    state = _load_base_state(path)
    model = _build_original_architecture()
    # The upstream loader filters detector.* and uses strict=False, which would
    # silently skip this checkpoint's unprefixed model weights. Do not use it.
    if checkpoint_kind == "demo":
        _configure_native_grid(model)
    model.load_state_dict(state, strict=True)
    del state
    if checkpoint_kind == "base":
        _configure_native_grid(model)
    model.to(torch.device(device))
    model.train(training)
    package_root = Path(_import_upstream().__file__).resolve().parent
    provenance_files = (
        "model_builder.py", "model/vitdet.py", "model/decoder.py",
        "model/maskformer_segmentation.py", "model/sam3_image.py",
    )
    model.demo512_metadata = {
        "loaded_checkpoint": str(path),
        "loaded_checkpoint_sha256": actual_hash,
        "checkpoint_kind": checkpoint_kind,
        "base_sha256": BASE_SHA256,
        "checkpoint_loaded_strictly": True,
        "base_loaded_strictly": checkpoint_kind == "base",
        "sam3_version": "0.1.0",
        "sam3_source": str(package_root),
        "sam3_source_sha256": {
            name: checkpoint_sha256(package_root / name) for name in provenance_files
        },
        "class_prompts": dict(CLASS_PROMPTS),
        "image_tensor_shape": ["B", 3, IMAGE_SIZE, IMAGE_SIZE],
        "image_normalization": "float RGB [0,1] -> (image - 0.5) / 0.5",
        "image_resize": False,
        "rgb_canvas": False,
        "patch_kernel_stride": PATCH_SIZE,
        "patch_internal_padding_each_side": PATCH_PADDING,
        "patch_token_grid": [TOKEN_GRID, TOKEN_GRID],
        "global_rope_grid": [TOKEN_GRID, TOKEN_GRID],
        "local_rope_grid": [24, 24],
        "local_window_token_padding_grid": [48, 48],
        "fpn_grids": [148, 74, 37],
        "raw_semantic_grid": [SEMANTIC_GRID, SEMANTIC_GRID],
        "logit_postprocess": "bilinear148->518, crop rows/cols[3:515]",
        "logit_source": "SAM3 UniversalSegmentationHead.semantic_seg_head",
        "legacy_instance_union": False,
        "learned_parameter_shapes_changed": False,
        "vit_mlp_execution": "original layers, unfused differentiable linear+GELU",
    }
    return model


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected RGB [B,3,512,512], got {tuple(images.shape)}")
    if images.shape[0] == 0 or not images.is_floating_point():
        raise ValueError("Images must be a nonempty floating-point RGB batch")
    if not bool(torch.isfinite(images).all()) or bool(
        (images.min() < 0) | (images.max() > 1)
    ):
        raise ValueError("Images must contain finite RGB values in [0,1]")


def forward_class_outputs(
    model: nn.Module, images: torch.Tensor, class_name: str
) -> dict[str, Any]:
    """Differentiable original SAM3 grounding outputs, before mask thresholding.

    The original image, text, geometry, fusion encoder, query decoder, scoring,
    and segmentation modules all run. This is the prediction part of upstream
    forward_grounding; its target matching is intentionally excluded. Returned
    ``semantic_seg`` and ``pred_masks`` remain on the raw 148x148 logit grid.
    ``scores`` reproduces the processor's instance probability times presence.
    Neither instance confidence filtering nor semantic thresholding is applied.
    """
    _validate_images(images)
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be a nonempty SAM3 text prompt")
    class_name = class_name.strip()
    if class_name.upper() in CLASS_PROMPTS:
        class_name = CLASS_PROMPTS[class_name.upper()]
    elif class_name.lower() in CLASS_PROMPTS.values():
        class_name = class_name.lower()
    else:
        raise ValueError(f"Unknown class {class_name!r}; use CRC, DLM, or SPL")
    if not hasattr(model, "demo512_metadata"):
        raise ValueError("Use build_demo_model to configure the native-512 geometry")
    from sam3.model.data_misc import FindStage

    device = next(model.parameters()).device
    if images.device != device:
        raise ValueError(f"Images are on {images.device}, but model is on {device}")
    batch = images.shape[0]
    normalized = (images - 0.5) / 0.5
    # This is the first RGB backbone input: exactly [B,3,512,512].
    backbone_out = model.backbone.forward_image(normalized)
    backbone_out.update(model.backbone.forward_text([class_name], device=device))
    find_input = FindStage(
        img_ids=torch.arange(batch, device=device, dtype=torch.long),
        text_ids=torch.zeros(batch, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    geometric_prompt = model._get_dummy_prompt(num_prompts=batch)
    prompt, prompt_mask, backbone_out = model._encode_prompt(
        backbone_out, find_input, geometric_prompt
    )
    backbone_out, encoder_out, _ = model._run_encoder(
        backbone_out, find_input, prompt, prompt_mask
    )
    out, hs = model._run_decoder(
        memory=encoder_out["encoder_hidden_states"],
        pos_embed=encoder_out["pos_embed"],
        src_mask=encoder_out["padding_mask"],
        out={},
        prompt=prompt,
        prompt_mask=prompt_mask,
        encoder_out=encoder_out,
    )
    model._run_segmentation_heads(
        out=out,
        backbone_out=backbone_out,
        img_ids=find_input.img_ids,
        vis_feat_sizes=encoder_out["vis_feat_sizes"],
        encoder_hidden_states=encoder_out["encoder_hidden_states"],
        prompt=prompt,
        prompt_mask=prompt_mask,
        hs=hs,
    )
    out["scores"] = out["pred_logits"].sigmoid() * out[
        "presence_logit_dec"
    ].sigmoid().unsqueeze(1)
    return out


def logits_to_native(logits: torch.Tensor) -> torch.Tensor:
    """Map raw SAM3 mask logits to original pixels, accounting for patch padding."""
    if logits.ndim != 4 or logits.shape[-2:] != (SEMANTIC_GRID, SEMANTIC_GRID):
        raise ValueError(f"Expected [B,C,148,148] mask logits, got {tuple(logits.shape)}")
    padded = F.interpolate(
        logits.float(),
        size=(PADDED_FIELD_SIZE, PADDED_FIELD_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return padded[
        ..., PATCH_PADDING : PATCH_PADDING + IMAGE_SIZE,
        PATCH_PADDING : PATCH_PADDING + IMAGE_SIZE,
    ]


def forward_class_logits(
    model: nn.Module, images: torch.Tensor, class_name: str
) -> torch.Tensor:
    """Return existing text-conditioned semantic logits as [B,1,512,512]."""
    outputs = forward_class_outputs(model, images, class_name)
    logits = logits_to_native(outputs["semantic_seg"])
    if logits.shape != (images.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"Unexpected native semantic shape: {tuple(logits.shape)}")
    return logits
