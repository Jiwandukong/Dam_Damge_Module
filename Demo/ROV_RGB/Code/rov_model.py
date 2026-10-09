"""DeepLabV3+ inference with the supplied Crack/spalling demonstration model."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import segmentation_models_pytorch as smp

def load_model(checkpoint,device):
    checkpoint=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
    cfg=checkpoint['cfg']
    if cfg['classes']!=['Background','Crack','박락']:
        raise ValueError('Expected Background / Crack / 박락 checkpoint channels')
    spec=cfg.get('model',{})
    if spec.get('name','deeplabv3plus')!='deeplabv3plus':raise ValueError('Expected DeepLabV3+')
    model=smp.DeepLabV3Plus(encoder_name=spec.get('encoder','resnet101'),encoder_weights=None,in_channels=3,classes=3)
    model.load_state_dict(checkpoint['model'],strict=True);model.to(device).eval()
    width=cfg.get('image_width',cfg['image_size']);height=cfg.get('image_height',cfg['image_size'])
    return model,(int(width),int(height)),cfg

def predict(model,rgb,size,device):
    h,w=rgb.shape[:2];image=cv2.resize(rgb,size,interpolation=cv2.INTER_LINEAR)
    x=torch.from_numpy(image.transpose(2,0,1).copy()).float().div_(255).unsqueeze(0).to(device)
    with torch.inference_mode():
        if torch.device(device).type=='cuda':
            with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
        else:logits=model(x)
        native=F.interpolate(logits.float(),size=(h,w),mode='bilinear',align_corners=False)
        mask=native.argmax(1)[0].byte().cpu().numpy()
        probability=native.softmax(1)[0].cpu().numpy()
    if probability.shape!=(3,h,w) or not np.isfinite(probability).all():raise ValueError('Invalid prediction')
    return mask,probability
