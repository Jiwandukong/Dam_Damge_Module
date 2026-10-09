#!/usr/bin/env python3
"""Save a standalone figure of spillway waterline grids and grouped frames."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from paths import ROOT, WORK, read_json
from demo_mapping import SpillwayWaterlineGridSampler


def main():
    summary = read_json(WORK/'inference_summary.json')
    if summary['mapping_status'] != 'demo_random':
        raise ValueError('Run the demonstration mapping before generating this figure')
    meta = summary['demo_mapping']
    sampler = SpillwayWaterlineGridSampler(meta['model'], seed=meta['seed'],
                                   clearance=meta['minimum_submergence_m'])
    rows = []
    for kind in ['CRC', 'SPL']:
        rows += list(csv.DictReader((ROOT/'Output/Result'/f'{kind}_result.csv').open(encoding='utf-8-sig')))
    walls = np.array([fragment[:, :3] for _, _, _, fragment in sampler.fragments])
    origin = walls.reshape(-1, 3).mean(0); origin[2] = 0
    walls -= origin
    water = np.array([w['triangle'] for w in sampler.water])-origin
    fig = plt.figure(figsize=(15, 9), layout='constrained')
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.add_collection3d(Poly3DCollection(water, facecolor='#33a6d0', edgecolor='#1588af',
                                       linewidth=.6, alpha=.08, zorder=1))
    ax.add_collection3d(Poly3DCollection(walls, facecolor='#888b8e', edgecolor='#666b70',
                                       linewidth=.2, alpha=.25, zorder=2))
    for kind, color, marker, label in [('CRC', '#ed2c35', 'o', 'Crack'), ('SPL', '#ffe22c', '^', 'Spalling')]:
        selected = [r for r in rows if r['damage_type']==kind]
        points = np.array([[float(r[k]) for k in ['world_center_x_m','world_center_y_m','world_center_z_m']]
                           for r in selected])-origin
        ax.scatter(points[:,0], points[:,1], points[:,2], c=color, marker=marker, s=44,
                   edgecolor='#303030', linewidth=.4, depthshade=False, zorder=5,
                   label=f'{label}: {len(selected)} observations')
    all_points = walls.reshape(-1,3)
    low, high = all_points.min(0), all_points.max(0)
    ax.set_xlim(low[0]-5,high[0]+5); ax.set_ylim(low[1]-5,high[1]+5); ax.set_zlim(low[2]-2,high[2]+4)
    ax.set_box_aspect(np.maximum(high-low,1))
    ax.view_init(elev=28, azim=25)
    ax.set_xlabel(f'Easting offset (m)\norigin {origin[0]:.2f}'); ax.set_ylabel(f'Northing offset (m)\norigin {origin[1]:.2f}')
    ax.set_zlabel('Model elevation (m)')
    ax.legend(loc='upper right')
    ax.set_title('ROV demonstration: similar frames along spillway waterline grids\n'
                 f"Synthetic locations; downstream water in blue; frame step {meta['frame_step_m']:.2f} m", pad=18)
    destination = WORK/'Reports/demo_random_3d_20261009.png'
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=160)
    plt.close(fig)
    print(destination)


if __name__=='__main__':
    main()
