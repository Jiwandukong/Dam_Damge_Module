"""Geometry/autograd regression tests; full 840M-parameter CUDA smoke is manual."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo512.model import (
    IMAGE_SIZE,
    PATCH_PADDING,
    TOKEN_GRID,
    _configure_native_grid,
    _import_upstream,
    _validate_images,
    logits_to_native,
)


class NativeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_patch_convolution_reaches_every_pixel_including_last_row_column(self):
        projection = nn.Conv2d(3, 1, 14, stride=14, padding=PATCH_PADDING, bias=False)
        nn.init.ones_(projection.weight)
        images = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True)
        seen = []
        handle = projection.register_forward_pre_hook(
            lambda module, args: seen.append(tuple(args[0].shape))
        )
        tokens = projection(images)
        tokens.sum().backward()
        handle.remove()
        self.assertEqual(seen, [(1, 3, 512, 512)])
        self.assertEqual(tuple(tokens.shape), (1, 1, TOKEN_GRID, TOKEN_GRID))
        # Every real pixel contributes to exactly one patch, including edges.
        self.assertTrue(torch.equal(images.grad, torch.ones_like(images)))

    def test_output_postprocessing_preserves_original_pixel_coordinates(self):
        # 148 raw cells cover 518 padded pixels. Encode original x coordinates
        # at their cell centers, including the -3 pixel padding offset.
        x_centers = (torch.arange(148, dtype=torch.float32) + 0.5) * 3.5 - 0.5 - 3
        raw = x_centers.view(1, 1, 1, 148).expand(1, 1, 148, 148).clone()
        raw.requires_grad_()
        native = logits_to_native(raw)
        expected = torch.arange(512, dtype=torch.float32).view(1, 1, 1, 512)
        self.assertEqual(tuple(native.shape), (1, 1, 512, 512))
        torch.testing.assert_close(native, expected.expand_as(native), atol=1e-4, rtol=0)
        native[..., -1, -1].backward()
        self.assertGreater(raw.grad[..., -1, -1].item(), 0)

    def test_native_geometry_keeps_parameters_and_backpropagates_through_vit(self):
        _import_upstream()
        from sam3.model.vitdet import ViT
        from sam3.model.position_encoding import PositionEmbeddingSine

        # Exercise real upstream local/global attention and its native geometry
        # on a small-width trunk, without loading the full base for unit tests.
        trunk = ViT(
            img_size=1008,
            pretrain_img_size=336,
            patch_size=14,
            embed_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2,
            use_abs_pos=True,
            tile_abs_pos=True,
            global_att_blocks=(1,),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=24,
            pretrain_use_cls_token=True,
            retain_cls_token=False,
            bias_patch_embed=False,
        )
        model = nn.Module()
        model.backbone = nn.Module()
        model.backbone.vision_backbone = nn.Module()
        model.backbone.vision_backbone.trunk = trunk
        model.backbone.vision_backbone.position_encoding = PositionEmbeddingSine(32)
        model.transformer = nn.Module()
        model.transformer.decoder = nn.Module()
        model.transformer.decoder.coord_cache = {}
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        _configure_native_grid(model)
        self.assertEqual(trunk.blocks[0].attn.freqs_cis.shape[0], 24 * 24)
        self.assertEqual(trunk.blocks[1].attn.freqs_cis.shape[0], 37 * 37)
        for name, param in model.named_parameters():
            self.assertTrue(torch.equal(param, before[name]), name)
        images = torch.randn(1, 3, 512, 512, requires_grad=True)
        out = trunk(images)[-1]
        self.assertEqual(tuple(out.shape[-2:]), (37, 37))
        out.square().mean().backward()
        self.assertTrue(bool(images.grad.isfinite().all()))
        self.assertGreater(images.grad[..., -1, -1].abs().sum().item(), 0)
        for block in trunk.blocks:
            self.assertIsNotNone(block.mlp.fc1.weight.grad)

    def test_rejects_resized_or_invalid_rgb_input(self):
        for image in (
            torch.zeros(1, 3, 1008, 1008),
            torch.zeros(1, 3, 518, 518),
            torch.zeros(1, 1, 512, 512),
            torch.zeros(1, 3, 512, 512, dtype=torch.uint8),
            torch.full((1, 3, 512, 512), float("nan")),
            torch.full((1, 3, 512, 512), -1.0),
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaises(ValueError):
                    _validate_images(image)


if __name__ == "__main__":
    unittest.main()
