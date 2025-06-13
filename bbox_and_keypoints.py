## Detect bounding boxes and keypoints on BOP dataset

import json
import cv2
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torchvision import transforms as T
from models import FRCNN, StackedHourglass, fasterrcnn_backbone
from bop_dataset import BOPDataset
from poseOpt import pose_coordinate_descend

from train.transforms import ToTensor, Normalize, AffineCrop, Normalize_imgnet
from misc.pose2d_eval import Pose2DEval
from bop_toolkit_lib.inout import save_bop_results

def one_each(pred, thresh=0.0):
    # Postprocess frcnn: get at most one instance per class
    # Return: boxes and labels
    conf = pred['scores'] > thresh

    conf_scores = pred['scores'][conf]
    conf_boxes = pred['boxes'][conf].int()
    conf_labels = pred['labels'][conf].int()

    valid = torch.zeros_like(conf_labels).bool()
    unique_labels = torch.unique(conf_labels)
    for uni in unique_labels:
        p = (conf_labels==uni).nonzero(as_tuple=False).reshape(-1)
        valid[p[0]] = True

    pd_scores = conf_scores[valid]
    pd_boxes = conf_boxes[valid]
    pd_labels = conf_labels[valid]
    
    return pd_boxes, pd_labels

#*********************************************************
#  Provide dataset name
#*********************************************************
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, help='dataset name')
parser.add_argument('--split', type=str, default="test", help='dataset name')
args = parser.parse_args()
dataset_name = args.dataset


#*********************************************************
#  Keypoint-based 6DOF estimation
#*********************************************************
root = './data/bop'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running on:', dataset_name)
print('Device:', device)

# Load dataset meta 
dataset = BOPDataset(root, dataset_name, split=args.split, 
                     return_keypoints=False, return_coco=True)
dataset._set_kpts_info()

num_classes = {'lmo':8, 'ycbv': 21, 'tudl': 3} 

# Load Faster-RCNN detector
detector_trainsform = T.ToTensor()
state_dict = torch.load('data/detect_checkpoints/d{}.pt'.format(dataset_name), map_location=device)['frcnn']

detector = fasterrcnn_backbone('resnet101', num_classes=1+num_classes[dataset_name]).to(device)
detector.eval()
detector.load_state_dict(state_dict)


# Load keypoint detector: stacked hourglass
transform_list = []
transform_list.append(AffineCrop(out_size=256, scale_factor=0, rotation_factor=0, dialation=0.25))
transform_list.append(ToTensor())
transform_list.append(Normalize())
kpts_transform = T.Compose(transform_list)
state_dict = torch.load('data/kpts_checkpoints/{}.pt'.format(dataset_name), map_location=device)['stacked_hg']

kpts_detector = StackedHourglass(dataset.n_kpts).to(device)
kpts_detector.eval()
kpts_detector.load_state_dict(state_dict)

# Run keypoint-based 6DOF
db = dataset.db
num_imgs = len(db)
poseEval = Pose2DEval()

obj2idx = dataset.obj2idx
idx2obj = {v:k for k,v in obj2idx.items()}
lab2obj = {v+1:k for k,v in obj2idx.items()}

with open('kpts3d.json', 'r') as infile:
    kpts3d = json.load(infile)[dataset.dataset_name]

results = {}
for i in tqdm(range(num_imgs)):
    imgpath = db[i]['imgpath']
    image = dataset.load_img(imgpath)
    
    scene_id = db[i]['scene_id']
    im_id = db[i]['im_id']
    K = db[i]['K']
    gt_objs = [lab2obj[l] for l in db[i]['labels']]

    results[im_id] = {}
    
    # Object detection
    with torch.no_grad():
        img = detector_trainsform(image).to(device)
        pred = detector([img])[0]
        pred = {k:v.cpu() for k,v in pred.items()}
        
    pd_boxes, pd_labels = one_each(pred, thresh=0)
    pd_objs = [lab2obj[i] for i in pd_labels.tolist()]
    pd_objs = torch.tensor(pd_objs)
    
    # Keypoint-base 6DOF estimation
    for obj in gt_objs:
        ### If Object is not detected
        if obj not in pd_objs:
            continue
        
        ### If Object is detected
        box = pd_boxes[pd_objs == obj].squeeze().tolist()
        box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        input_crop = {'image':image, 'bb':box}
        input_crop = kpts_transform(input_crop)
        
        with torch.no_grad():
            batch = input_crop['image'][None].to(device)
            output = kpts_detector(batch)
            output = output[-1].cpu()
            
            kpt_start = dataset.obj2kptid[obj][0]
            kpt_end = dataset.obj2kptid[obj][1]
            heatmaps_pred = output[[0], kpt_start:kpt_end, :, :]
       
            kpts_pred, confs = poseEval.heatmaps_to_locs(heatmaps_pred, return_vals=True)
            confs = confs[0]
            kpts_pred = kpts_pred[0]
            
            crop_kpts = kpts_pred * (256/64)
            view_kpts = poseEval.get_view_kpts(box, crop_kpts)
            view_kpts = view_kpts.numpy()
            kpts_conf = np.hstack([view_kpts, confs[:,None]])

            # save
            results[im_id][obj] = kpts_conf.T.tolist()



with open(f"detections_{dataset_name}.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)


# FORMAT:
# 2 files per dataset
# 1) detections
# - img_id, dict() object_id: [u, v, conf]
# 2) data
# - kpts3d
# - camK


# FORMAT:
# 1 file per dataset
# Dict() with keys: 
# - 1,2,3,4,5 (object_ids): Dict() with keys:
#   - keypoints [2, N, num_frames], 0 indicates no detection
#   - conf [N, num_frames]
#   - kpts3d [3,N]
# - cam_K: camera calibration matrix [3, 3]
# - img_id: image numbers [num_frames]
# will keep gt pose separate!

# I want:
# FOR POSE ESTIMATION
# 1) keypoint pixel measurements (directly from detector)
# 2) 3D keypoint library
# 3) camera calibration matrix (BOP)
# FOR EVALUATION
# 4) associated image number
# 5) gt pose