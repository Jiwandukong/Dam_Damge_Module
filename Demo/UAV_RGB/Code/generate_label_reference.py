#!/usr/bin/env python3
"""Export every reviewed label as an individual square, source-only crop."""

import csv
from collections import Counter
import hashlib
import html
import json
import math
import posixpath
from pathlib import Path

import PIL
from PIL import Image, ImageChops, ImageDraw


BUNDLE = Path(__file__).resolve().parents[1]
BASE_COLUMNS = [
    "image", "damage_id", "damage_type", "damage_name_ko", "pixel_nodes_json",
    "world_center_x_m", "world_center_y_m", "world_center_z_m",
    "length_px", "length_m", "width_px", "width_m", "area_m2",
]
COLUMNS = BASE_COLUMNS + [
    "source_image_path", "crop_original_path", "overlay_path",
    "crop_x0_px", "crop_y0_px", "crop_width_px", "crop_height_px",
]
OVERLAY_COLOR = (255, 0, 0)
OVERLAY_ALPHA = 0.30
CRACK_BUFFER_RADIUS_PX = 2
CROP_MARGIN_RATIO = 0.20
CROP_MIN_MARGIN_PX = 128


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rasterize(size, shape):
    points = [(math.floor(x + 0.5), math.floor(y + 0.5))
              for x, y in shape["points"]]
    if any(not (0 <= x < size[0] and 0 <= y < size[1]) for x, y in points):
        raise ValueError("A label point lies outside its original photograph")
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if shape["label"] == "CRC" and shape["shape_type"] == "linestrip":
        draw.line(points, fill=255, width=1)
    elif shape["label"] in ("DLM", "SPL") and shape["shape_type"] == "polygon":
        draw.polygon(points, fill=255)
    else:
        raise ValueError("Unexpected damage class or annotation geometry")
    return mask


def visualization_mask(mask, kind, radius=CRACK_BUFFER_RADIUS_PX):
    """Buffer CRC for display using a pixel disk, preserving the label mask."""
    if kind != "CRC" or radius == 0:
        return mask.copy()
    bbox = mask.getbbox()
    if bbox is None:
        return mask.copy()
    region = (max(0, bbox[0] - radius), max(0, bbox[1] - radius),
              min(mask.width, bbox[2] + radius), min(mask.height, bbox[3] + radius))
    source = mask.crop(region)
    expanded = source.copy()
    width, height = source.size
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius or (dx == 0 and dy == 0):
                continue
            source_box = (max(0, -dx), max(0, -dy), min(width, width - dx), min(height, height - dy))
            if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
                continue
            shifted = Image.new("L", source.size, 0)
            shifted.paste(source.crop(source_box), (max(0, dx), max(0, dy)))
            expanded = ImageChops.lighter(expanded, shifted)
    result = Image.new("L", mask.size, 0)
    result.paste(expanded, region[:2])
    return result


def render_overlay(original, display_mask):
    painted = original.copy()
    painted.paste(OVERLAY_COLOR, (0, 0), display_mask)
    return Image.blend(original, painted, OVERLAY_ALPHA)


def crop_bounds(bbox, size, step=128, margin_ratio=CROP_MARGIN_RATIO,
                min_margin=CROP_MIN_MARGIN_PX, allow_limited_context=False):
    """Square crop with genuine photographic context on all four sides."""
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    margin = max(min_margin, math.ceil(max(width, height) * margin_ratio))
    side = math.ceil((max(width, height) + 2 * margin) / step) * step
    if side > min(size):
        if not allow_limited_context:
            raise ValueError("The source photograph lacks enough real context for this square crop")
        side = (min(size) // step) * step
        if side < max(width, height):
            raise ValueError("A source-only square cannot contain the entire label without padding or resizing")
    x0 = max(0, min(math.floor((left + right - side) / 2), size[0] - side))
    y0 = max(0, min(math.floor((top + bottom - side) / 2), size[1] - side))
    bounds = (x0, y0, x0 + side, y0 + side)
    if not (x0 <= left < right <= bounds[2] and y0 <= top < bottom <= bounds[3]):
        raise ValueError("Computed crop does not contain the complete damage")
    if (not allow_limited_context and
            min(left - x0, top - y0, bounds[2] - right, bounds[3] - bottom) < margin):
        raise ValueError("The source photograph lacks enough real context around this damage")
    return bounds, margin


def source_crop(image, bounds):
    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError("Crop bounds must stay inside the original photograph; padding is disabled")
    return image.crop(bounds)


def generate(bundle=BUNDLE, output=None):
    bundle = Path(bundle)
    if PIL.__version__ != "12.2.0":
        raise RuntimeError("Install the pinned Pillow==12.2.0 to preserve published label rasterization")
    manifest = json.loads((bundle / "Code/label_reference.json").read_text())
    output = Path(output) if output is not None else bundle / "Output"
    output.mkdir(exist_ok=False)
    rows, connections, seen = [], [], set()
    photo_path = None
    photo = None
    verified_sources = {}
    for record in manifest["records"]:
        row = {key: record["row"][key] for key in BASE_COLUMNS}
        damage_id, kind = row["damage_id"], row["damage_type"]
        if damage_id in seen or record["shape"]["label"] != kind:
            raise ValueError("Duplicate damage ID or class mismatch")
        seen.add(damage_id)
        source = bundle / "Rawdata/images" / row["image"]
        if source not in verified_sources:
            verified_sources[source] = sha256(source)
        if verified_sources[source] != record["source_image_sha256"]:
            raise ValueError("The original GitHub photograph has changed")
        if source != photo_path:
            with Image.open(source) as opened:
                photo = opened.convert("RGB")
            photo_path = source
        if photo.size != tuple(record["image_size_px"]):
            raise ValueError("Source photograph dimensions do not match the label")
        mask = rasterize(photo.size, record["shape"])
        original_area = mask.histogram()[255]
        if original_area != record["published_label_area_px"]:
            raise ValueError("Label geometry differs from the reviewed published label")
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError("Empty damage geometry")
        full_display_mask = visualization_mask(mask, kind)
        display_bbox = full_display_mask.getbbox()
        bounds, margin = crop_bounds(display_bbox, photo.size, allow_limited_context=True)
        actual_context = {"left": display_bbox[0] - bounds[0], "top": display_bbox[1] - bounds[1],
                          "right": bounds[2] - display_bbox[2], "bottom": bounds[3] - display_bbox[3]}
        source_edges = [name for name, touches in (
            ("left", bbox[0] == 0), ("top", bbox[1] == 0),
            ("right", bbox[2] == photo.width), ("bottom", bbox[3] == photo.height)) if touches]
        context_sufficient = min(actual_context.values()) >= margin
        context_status = "source_edge" if source_edges else (
            "full_context" if context_sufficient else "limited_context")
        original = source_crop(photo, bounds)
        crop_mask = mask.crop(bounds)
        if crop_mask.histogram()[255] != original_area:
            raise ValueError("The damage was clipped during cropping")
        display_mask = full_display_mask.crop(bounds)
        display_area = full_display_mask.histogram()[255]
        if display_mask.histogram()[255] != display_area:
            raise ValueError("The visual crack buffer was clipped during cropping")
        overlay = render_overlay(original, display_mask)
        # Pixels outside the displayed damage/buffer must remain unchanged.
        changed = ImageChops.difference(overlay, original)
        for channel in changed.split():
            outside = ImageChops.multiply(channel, ImageChops.invert(display_mask))
            if outside.getbbox() is not None:
                raise ValueError("The overlay changed pixels outside this damage")
        original_path = f"original_crops/{kind}/{damage_id}.png"
        overlay_path = f"overlays/{kind}/{damage_id}.png"
        for relative, image in [(original_path, original), (overlay_path, overlay)]:
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        row.update(source_image_path=f"../Rawdata/images/{row['image']}",
                   crop_original_path=original_path, overlay_path=overlay_path,
                   crop_x0_px=bounds[0], crop_y0_px=bounds[1],
                   crop_width_px=original.width, crop_height_px=original.height)
        rows.append(row)
        connections.append({
            "damage_id": damage_id, "damage_type": kind,
            "source_annotation_id": record["source_annotation_id"],
            "shape_index": record["shape_index"],
            "source_image_sha256": record["source_image_sha256"],
            "label_bbox_xyxy_px": list(bbox), "crop_bbox_xyxy_px": list(bounds),
            "display_bbox_xyxy_px": list(display_bbox),
            "actual_context_px": actual_context,
            "context_sufficient": context_sufficient, "context_status": context_status,
            "source_edge_contacts": source_edges,
            "requested_margin_px": margin, "label_area_px": original_area,
            "display_area_px": display_area,
            "display_buffer_radius_px": CRACK_BUFFER_RADIUS_PX if kind == "CRC" else 0,
            "complete_damage_preserved": True,
            "previous_linked_tile_count": record["previous_linked_tile_count"],
            "source_image_path": row["source_image_path"],
            "crop_original_path": original_path, "overlay_path": overlay_path,
            "original_crop_sha256": sha256(output / original_path),
            "overlay_sha256": sha256(output / overlay_path),
        })
        if len(rows) % 20 == 0 or len(rows) == len(manifest["records"]):
            print(f"Generated {len(rows)}/{len(manifest['records'])}: {damage_id} {original.width}px square", flush=True)
    with (output / "damage_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for kind in ("CRC", "DLM", "SPL"):
        class_dir = output / kind
        class_dir.mkdir()
        class_rows = []
        for row in rows:
            if row["damage_type"] != kind:
                continue
            class_row = dict(row)
            for key in ("source_image_path", "crop_original_path", "overlay_path"):
                class_row[key] = posixpath.normpath("../" + class_row[key])
            class_rows.append(class_row)
        with (class_dir / "damage_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(class_rows)
    summary = {"total_damages": len(rows), "class_counts": dict(Counter(r["damage_type"] for r in rows)),
               "context_status_counts": dict(Counter(d["context_status"] for d in connections)),
               "source_images": len(verified_sources), "original_crop_images": len(rows), "overlay_images": len(rows)}
    metadata = {
        "schema_version": "damage-crop-v1", "path_base": "this_output_folder",
        "coordinate_contract": {"horizontal_crs": "EPSG:5186", "unit": "m", "vertical_axis": "Z-up"},
        "crop_policy": {"shape": "square", "step_px": 128,
                        "min_margin_px": CROP_MIN_MARGIN_PX, "margin_ratio": CROP_MARGIN_RATIO,
                        "source_resized": False, "padding": False,
                        "retain_all_reviewed_damage_ids": True,
                        "edge_handling": "shift inside original photograph; preserve every label pixel; flag limited context and source edges"},
        "overlay": {"alpha": OVERLAY_ALPHA, "transparency": 1 - OVERLAY_ALPHA,
                    "color_rgb": OVERLAY_COLOR,
                    "class_colors_rgb": {kind: OVERLAY_COLOR for kind in ("CRC", "DLM", "SPL")},
                    "scope": "selected_damage_only",
                    "crack_buffer": {"radius_px": CRACK_BUFFER_RADIUS_PX, "shape": "pixel_disk",
                                     "application": "visualization_only_original_geometry_and_measurements_preserved"}},
        "repository": manifest["repository"], "source_commit": manifest["source_commit"],
        "result_source": "reviewed_label_reference", "model_inference_used_for_this_bundle": False,
        "columns": COLUMNS, "damages": connections,
        "sample_replacements": manifest.get("sample_replacements", []),
        "summary": summary, "pillow_version": PIL.__version__,
        "published_overlay_audit": manifest.get("published_overlay_audit", {}),
    }
    (output / "connections.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate approved label-reference CSV and individual square overlays.")
    parser.add_argument("--output", type=Path, help="Fresh output directory; existing results are preserved.")
    args = parser.parse_args()
    generate(output=args.output)
