"""Original Sunflicker algorithm, using raw image targets and video neighbors."""
from pathlib import Path
import re
import cv2
from paths import ROOT,WORK,sha,read_json,write_json
from algorithms import online_sunflicker as algorithm

PARAMETERS=dict(window_radius=5,window_type='centered',align_method='ecc',channel='gray',lp_sigma=21.0,
                latent_dim=8,prediction_model='ar1',correction='bright_only_additive',correction_strength=.8,
                min_valid_observations=2,model_history=10,downsample_factor=8)

class Preprocessor:
    def __init__(self,video,work=WORK):
        self.video=Path(video);self.cache_dir=Path(work)/'sunflicker_cache';self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.algorithm_sha=sha(ROOT/'Code/algorithms/online_sunflicker.py')
        self.video_sha=sha(self.video) if self.video.exists() else None
        self.video_cache=None
    def close(self):
        if self.video_cache is not None:self.video_cache.close()
    def get(self,image_path):
        image_path=Path(image_path);raw_sha=sha(image_path)
        cache=self.cache_dir/(image_path.stem+'.png');meta=cache.with_suffix('.json')
        if cache.exists() and meta.exists():
            record=read_json(meta)
            if (record['image_sha256']==raw_sha and record['algorithm_sha256']==self.algorithm_sha
                and record['parameters']==PARAMETERS and (self.video_sha is None or record['video_sha256']==self.video_sha)
                and sha(cache)==record['processed_sha256']):
                image=cv2.imread(str(cache))
                if image is not None:return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),'verified_rgb_cache'
        if not self.video.exists():raise FileNotFoundError('Sunflicker needs the source video or a verified RGB cache: '+str(self.video))
        match=re.fullmatch(r'frame_(\d+)',image_path.stem)
        if not match:raise ValueError('Raw image name must identify its video frame: frame_XXXXXX.jpg')
        index=int(match.group(1));target=cv2.imread(str(image_path))
        if target is None:raise ValueError('Cannot decode '+str(image_path))
        if self.video_cache is None:self.video_cache=algorithm.FrameCache(self.video,max_items=64)
        cache_object=self.video_cache;count=int(cache_object.cap.get(cv2.CAP_PROP_FRAME_COUNT));h,w=target.shape[:2]
        if not 0<=index<count:raise ValueError('Frame index outside source video')
        class TargetCache:
            def get(self,n):return target.copy() if n==index else cache_object.get(n)
        cfg=algorithm.SunflickerConfig(project_root=ROOT,input_video=self.video,output_root=self.cache_dir,
                                     source_dataset_dir=image_path.parent,debug_subset_dir=image_path.parent,
                                     show_progress=False,**PARAMETERS)
        result=algorithm.process_target_frame(cfg,TargetCache(),index,count,w,h)
        output=result[0]
        if output is None or output.shape!=target.shape:raise ValueError('Sunflicker output differs from target geometry')
        if not cv2.imwrite(str(cache),output):raise RuntimeError('Cannot save RGB cache')
        write_json(meta,dict(image_sha256=raw_sha,video_sha256=self.video_sha,algorithm_sha256=self.algorithm_sha,
                            parameters=PARAMETERS,processed_sha256=sha(cache),source='original Sunflicker RGB computation',status=result[6]))
        return cv2.cvtColor(output,cv2.COLOR_BGR2RGB),'computed_from_raw_rgb'
