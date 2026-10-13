"""Render measured registered TLS points in the ROV_LiDAR overlay style."""
import hashlib

import runtime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import proj3d

BACKGROUND = '#080c14'
POINT_SIZE = 6
IMAGE_SIZE = (3840, 2160)
DPI = 300
BRIGHT_COLORS = LinearSegmentedColormap.from_list(
    'bright_damage', ['#30dfff', '#70ffca', '#f8ff72', '#ff9860'])


def fit_camera(figure, axes, span, cloud, context):
    """Uniformly fit the orthographic view without changing physical geometry."""
    initial_zoom = .95
    axes.set_box_aspect(span, zoom=initial_zoom)
    figure.canvas.draw()
    visible = np.concatenate([cloud, context]) if len(context) else cloud
    x, y, _ = proj3d.proj_transform(*visible.T, axes.get_proj())
    pixels = axes.transData.transform(np.column_stack([x, y]))
    center = np.array([np.mean(axes.get_xlim()), np.mean(axes.get_ylim()), np.mean(axes.get_zlim())])
    x0, y0, _ = proj3d.proj_transform(*center, axes.get_proj())
    pivot = axes.transData.transform([x0, y0])
    lower, upper = pixels.min(axis=0), pixels.max(axis=0)
    width, height = figure.canvas.get_width_height()
    margin = np.array([width, height])*.055
    available_low = pivot-margin
    available_high = np.array([width, height])-margin-pivot
    factors = np.concatenate([available_low/np.maximum(pivot-lower, 1),
                              available_high/np.maximum(upper-pivot, 1)])
    zoom = initial_zoom*min(float(factors.min()), 3.)
    axes.set_box_aspect(span, zoom=zoom)


def snapshot(path, damage_id, positions, source_ids, context_ids, depth, display_basis):
    """One 4K overlay, no annotations, axes, legend or colorbar."""
    seed = int.from_bytes(hashlib.sha256(damage_id.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    selected = source_ids if len(source_ids) <= 180000 else rng.choice(source_ids, 180000, replace=False)
    context_ids = context_ids[~np.isin(context_ids, source_ids)]
    if len(context_ids) > 24000:
        context_ids = rng.choice(context_ids, 24000, replace=False)
    basis = np.asarray(display_basis, dtype=float)
    if not np.allclose(basis@basis.T, np.eye(3), atol=1e-10):
        raise ValueError('Display basis must preserve distances')
    center = positions[source_ids].mean(axis=0)
    cloud = (positions[selected]-center)@basis.T
    context = (positions[context_ids]-center)@basis.T
    if len(context):
        lower, upper = np.percentile(context[:, 2], [.5, 99.5])
        context = context[(context[:, 2] >= lower) & (context[:, 2] <= upper)]
    shown = np.concatenate([cloud, context]) if len(context) else cloud
    limits = np.array([shown.min(axis=0), shown.max(axis=0)])
    span = np.maximum(limits[1]-limits[0], .01)
    margin = np.maximum(span*.06, .01)
    limits += np.array([-1, 1])[:, None]*margin
    span = limits[1]-limits[0]
    values = np.asarray(depth[selected], dtype=float)
    if not np.isfinite(cloud).all() or not np.isfinite(values).all():
        raise ValueError(f'{damage_id}: non-finite observed coordinates/depth')
    color_low, color_high = np.percentile(values, [1, 99])
    norm = Normalize(vmin=float(color_low), vmax=float(color_high)+1e-12, clip=True)
    figure = plt.figure(figsize=(IMAGE_SIZE[0]/DPI, IMAGE_SIZE[1]/DPI), dpi=DPI, facecolor=BACKGROUND)
    axes = figure.add_axes([0, 0, 1, 1], projection='3d', computed_zorder=False)
    axes.set_facecolor(BACKGROUND)
    if len(context):
        axes.scatter(*context.T, c='#99a9bc', s=POINT_SIZE, alpha=.65,
                     linewidths=0, depthshade=False, clip_on=False, zorder=1)
    axes.scatter(*cloud.T, c=values, cmap=BRIGHT_COLORS, norm=norm, s=POINT_SIZE,
                 alpha=1, linewidths=0, depthshade=False, clip_on=False, zorder=2)
    axes.set_xlim(limits[:, 0])
    axes.set_ylim(limits[:, 1])
    axes.set_zlim(limits[:, 2])
    axes.view_init(elev=32, azim=-35)
    axes.set_proj_type('ortho')
    axes.set_axis_off()
    fit_camera(figure, axes, span, cloud, context)
    figure.savefig(path, facecolor=BACKGROUND)
    plt.close(figure)
