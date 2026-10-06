"""Check complete-damage coverage and 128px sizes for difficult crop cases."""

import unittest

from PIL import Image

from generate_label_reference import crop_bounds, source_crop, render_overlay, visualization_mask


class CropBoundsTests(unittest.TestCase):
    def assert_complete(self, bbox, size):
        bounds, margin = crop_bounds(bbox, size)
        x0, y0, x1, y1 = bounds
        self.assertEqual((x1 - x0) % 128, 0)
        self.assertEqual((y1 - y0) % 128, 0)
        self.assertEqual(x1 - x0, y1 - y0)
        self.assertLessEqual(x0, bbox[0])
        self.assertLessEqual(y0, bbox[1])
        self.assertGreaterEqual(x1, bbox[2])
        self.assertGreaterEqual(y1, bbox[3])
        self.assertGreaterEqual(min(bbox[0] - x0, bbox[1] - y0,
                                    x1 - bbox[2], y1 - bbox[3]), margin)
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, size[0])
        self.assertLessEqual(y1, size[1])
        return bounds

    def test_small_damage_crossing_old_tile_boundary(self):
        self.assert_complete((961, 142, 1087, 154), (5280, 3956))

    def test_long_damage_grows_beyond_512(self):
        bounds = self.assert_complete((600, 1400, 2727, 2047), (5280, 3956))
        self.assertGreater(bounds[2] - bounds[0], 2127)

    def test_damage_on_photo_edges_is_rejected_without_fabricating_context(self):
        for bbox in [(0, 80, 100, 130), (100, 0, 130, 80),
                     (5200, 3000, 5280, 3020), (100, 3900, 140, 3956)]:
            with self.subTest(bbox=bbox):
                with self.assertRaisesRegex(ValueError, "context"):
                    crop_bounds(bbox, (5280, 3956))

    def test_near_photo_edges_shift_square_crop_and_keep_real_context(self):
        for bbox in [(128, 300, 180, 350), (500, 128, 550, 180),
                     (5100, 3000, 5152, 3050), (500, 3780, 550, 3828)]:
            with self.subTest(bbox=bbox):
                self.assert_complete(bbox, (5280, 3956))

    def test_full_photo_damage_cannot_supply_surrounding_context(self):
        photo = Image.new("RGB", (200, 100), (12, 34, 56))
        with self.assertRaisesRegex(ValueError, "context"):
            crop_bounds((0, 0, 200, 100), photo.size)

    def test_source_crop_never_pads_or_stretches(self):
        photo = Image.new("RGB", (200, 200), (12, 34, 56))
        self.assertEqual(source_crop(photo, (20, 30, 148, 158)).size, (128, 128))
        self.assertEqual(source_crop(photo, (20, 30, 148, 158)).tobytes(),
                         photo.crop((20, 30, 148, 158)).tobytes())
        with self.assertRaisesRegex(ValueError, "padding is disabled"):
            source_crop(photo, (-1, 0, 127, 128))

    def test_retaining_edge_label_preserves_all_pixels_and_square_shape(self):
        bounds, margin = crop_bounds((0, 2919, 2127, 3566), (5280, 3956),
                                     allow_limited_context=True)
        x0, y0, x1, y1 = bounds
        self.assertEqual(x1 - x0, y1 - y0)
        self.assertEqual((x1 - x0) % 128, 0)
        self.assertEqual(x0, 0)
        self.assertLessEqual(y0, 2919)
        self.assertGreaterEqual(x1, 2127)
        self.assertGreaterEqual(y1, 3566)
        self.assertLessEqual(y1, 3956)
        self.assertGreater(margin, 0)

    def test_requested_square_larger_than_photo_reduces_context_without_cutting_label(self):
        bounds, _ = crop_bounds((100, 100, 400, 450), (640, 512),
                                allow_limited_context=True)
        self.assertEqual(bounds[2] - bounds[0], 512)
        self.assertEqual(bounds[3] - bounds[1], 512)
        self.assertLessEqual(bounds[0], 100)
        self.assertLessEqual(bounds[1], 100)
        self.assertGreaterEqual(bounds[2], 400)
        self.assertGreaterEqual(bounds[3], 450)


class OverlayTests(unittest.TestCase):
    def test_crack_disk_buffer_preserves_original_label(self):
        label = Image.new("L", (11, 11))
        label.putpixel((5, 5), 255)
        displayed = visualization_mask(label, "CRC")
        self.assertEqual(label.histogram()[255], 1)
        self.assertEqual(displayed.histogram()[255], 13)
        self.assertEqual(displayed.getpixel((5, 3)), 255)
        self.assertEqual(displayed.getpixel((4, 4)), 255)
        self.assertEqual(displayed.getpixel((3, 3)), 0)

    def test_crack_buffer_at_photo_edge_does_not_wrap(self):
        label = Image.new("L", (11, 11))
        label.putpixel((0, 0), 255)
        displayed = visualization_mask(label, "CRC")
        self.assertEqual(displayed.histogram()[255], 6)
        self.assertEqual(displayed.getpixel((10, 0)), 0)
        self.assertEqual(displayed.getpixel((0, 10)), 0)

    def test_area_damage_geometry_is_not_buffered(self):
        label = Image.new("L", (11, 11))
        label.putpixel((5, 5), 255)
        for kind in ("DLM", "SPL"):
            self.assertEqual(visualization_mask(label, kind).tobytes(), label.tobytes())

    def test_red_overlay_uses_30_percent_opacity_and_keeps_background(self):
        original = Image.new("RGB", (3, 3), (100, 150, 200))
        mask = Image.new("L", (3, 3))
        mask.putpixel((1, 1), 255)
        overlay = render_overlay(original, mask)
        self.assertEqual(overlay.getpixel((1, 1)), (146, 105, 140))
        self.assertEqual(overlay.getpixel((0, 0)), (100, 150, 200))


if __name__ == "__main__":
    unittest.main()
