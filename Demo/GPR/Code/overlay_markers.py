"""Render visible red crosses without resizing the pixel-aligned B-scan."""

from PIL import ImageDraw

DEFAULT_MARKER_HALF_SIZE = 10
DEFAULT_MARKER_WIDTH = 3
LONG_IMAGE_ASPECT_RATIO = 4.0
MARKER_REFERENCE_WIDTH = 1000


def marker_settings(image_width, half_size=DEFAULT_MARKER_HALF_SIZE,
                    line_width=DEFAULT_MARKER_WIDTH, *, image_height=512):
    if half_size < 1 or line_width < 1:
        raise ValueError("Marker half-size and width must be positive")
    # Long B-scans are heavily reduced in the viewer. Give their crosses a
    # visible screen size while retaining the existing markers on short scans.
    if image_height < 1:
        raise ValueError("Image height must be positive")
    is_long = image_width / image_height >= LONG_IMAGE_ASPECT_RATIO
    scale = max(1.0, image_width / MARKER_REFERENCE_WIDTH) if is_long else 1.0
    return max(1, round(half_size * scale)), max(1, round(line_width * scale))


def render_markers(base_image, centers, half_size=DEFAULT_MARKER_HALF_SIZE,
                   line_width=DEFAULT_MARKER_WIDTH):
    image = base_image.convert("RGB")
    half, width = marker_settings(image.width, half_size, line_width, image_height=image.height)
    draw = ImageDraw.Draw(image)
    for x, y in centers:
        cx, cy = round(float(x)), round(float(y))
        draw.line([(cx - half, cy), (cx + half, cy)], fill=(255, 0, 0), width=width)
        draw.line([(cx, cy - half), (cx, cy + half)], fill=(255, 0, 0), width=width)
    return image
