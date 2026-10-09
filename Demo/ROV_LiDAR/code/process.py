"""Generate final SC/DP LAS, 4K PNG and summary CSV without verification artifacts."""
import sys
sys.dont_write_bytecode = True
from runtime import ROOT, WORK
from pathlib import Path
import argparse
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['export','all','scour','slab','slab-extraction','slab-depression'],default='export',
      help='export uses retained analysis; all recomputes analysis before exporting')
    p.add_argument('--work-dir',type=Path,default=WORK)
    p.add_argument('--min-area',type=float,default=5.0,
      help='Minimum scour area in m²; depression output has no area filter')
    p.add_argument('--segmented',type=Path,default=ROOT/'data/01_EYAS_translated_segmented.ply')
    p.add_argument('--slab-zone',type=Path,default=ROOT/'data/01_EYAS_translated_segmented_slab_zone.ply')
    p.add_argument('--slab-mode',choices=['input-zone','auto'],default='input-zone',help='Retain supplied XY ROI or use original automatic slab extraction')
    p.add_argument('--slab-upper-z',type=float,default=42.708,help='Upper Z cap for this survey in input-zone mode; low observations are retained')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output')
    p.add_argument('--damage-type',choices=['all','SC','DP'],help='Export type; defaults to DP for slab tasks and SC for scour')
    p.add_argument('--dam-model',type=Path,default=ROOT.parent/'Dam_model/Daecheongdam/daecheongdam_regions_grid5m.gltf')
    p.add_argument('--cube-python',type=Path,default=Path(sys.executable))
    p.add_argument('--whitebox-dir',type=Path,default=ROOT/'code/tools/whitebox')
    args=p.parse_args()
    if args.min_area<0:p.error('--min-area must be nonnegative')
    work=args.work_dir.expanduser().resolve()
    if work==ROOT or ROOT in work.parents:p.error('--work-dir must be outside ROV_LiDAR')
    output=args.output_dir.expanduser().resolve()
    if work==output or output in work.parents or work in output.parents:
        p.error('--work-dir and --output-dir must be separate directories, without nesting')
    if args.task!='export':
        subprocess.run([sys.executable,'-B',str(ROOT/'code/run.py'),'--task',args.task,
          '--work-dir',str(work),'--segmented',str(args.segmented),'--slab-zone',str(args.slab_zone),
          '--slab-mode',args.slab_mode,'--slab-upper-z',str(args.slab_upper_z),
          '--cube-python',str(args.cube_python),'--whitebox-dir',str(args.whitebox_dir)],check=True)
    if args.task=='slab-extraction':return
    damage_type=args.damage_type or ('DP' if args.task in {'slab','slab-depression'} else 'SC' if args.task=='scour' else 'all')
    subprocess.run([sys.executable,'-B',str(ROOT/'code/export_results.py'),'--analysis-dir',str(work/'analysis'),
      '--source-ply',str(args.segmented),'--slab-zone',str(args.slab_zone),'--output-dir',str(args.output_dir),'--min-area',str(args.min_area),'--damage-type',damage_type,
      '--dam-model',str(args.dam_model),'--work-dir',str(work)],check=True)

if __name__=='__main__':main()
