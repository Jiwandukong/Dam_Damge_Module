"""Run the retained analytical stages, keeping intermediate outputs in the cache."""
import sys
sys.dont_write_bytecode = True
from runtime import ROOT, WORK
from pathlib import Path
import argparse
import subprocess

def run(args):
    analysis = args.work_dir/'analysis'
    jobs = []
    if args.task in {'all', 'scour'}:
        jobs.append(('scour.py', ['--input', str(args.segmented), '--output-dir', str(analysis/'scour'),
          '--cube-python', str(args.cube_python), '--whitebox-dir', str(args.whitebox_dir)]))
    if args.task in {'all', 'slab', 'slab-extraction'}:
        if args.slab_mode=='input-zone':
            jobs.append(('../prepare_slab_zone.py', ['--input',str(args.slab_zone),
              '--output-dir',str(analysis/'slab_depression/slab_extraction'),
              '--upper-z',str(args.slab_upper_z),'--work-dir',str(args.work_dir)]))
        else:
            jobs.append(('slab_extraction.py', ['--input', str(args.slab_zone),
              '--output-dir', str(analysis/'slab_depression/slab_extraction')]))
    if args.task in {'all', 'slab', 'slab-depression'}:
        jobs.append(('slab_depression.py', ['--input', str(analysis/'slab_depression/slab_extraction/slab_extracted.las'),
          '--output-dir', str(analysis/'slab_depression/depression')]))
    for name, arguments in jobs:
        print('START', name, flush=True)
        subprocess.run([sys.executable, '-B', str(ROOT/'code/algorithms'/name), *arguments], check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['all','scour','slab','slab-extraction','slab-depression'],default='all')
    p.add_argument('--segmented',type=Path,default=ROOT/'data/01_EYAS_translated_segmented.ply')
    p.add_argument('--slab-zone',type=Path,default=ROOT/'data/01_EYAS_translated_segmented_slab_zone.ply')
    p.add_argument('--slab-mode',choices=['input-zone','auto'],default='input-zone',help='Retain supplied XY ROI or use original automatic slab extraction')
    p.add_argument('--slab-upper-z',type=float,default=42.708,help='Survey-specific upper Z cap for input-zone mode; low points are retained')
    p.add_argument('--work-dir',type=Path,default=WORK)
    p.add_argument('--cube-python',type=Path,default=Path(sys.executable))
    p.add_argument('--whitebox-dir',type=Path,default=ROOT/'code/tools/whitebox')
    args=p.parse_args();args.work_dir=args.work_dir.expanduser().resolve()
    if args.work_dir==ROOT or ROOT in args.work_dir.parents:raise ValueError('Work directory must be outside ROV_LiDAR.')
    run(args)

if __name__=='__main__':main()
