# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""

import argparse
import os
import sys
from pathlib import Path
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import PIL.Image

import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg11, ResNet18_Weights


from script.Dataset import generate_bins, DetectedObject
from library.Math import *
from library.Plotting import *
from script import Model, ClassAverages
from script.Model import ResNet, ResNet18, VGG11

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# model factory to choose model
model_factory = {
    'resnet': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    'resnet18': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    # 'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}

class Bbox:
    def __init__(self, box_2d, class_, track_id=-1):
        self.box_2d = box_2d
        self.detected_class = class_
        self.track_id = track_id

def detect3d(
    reg_weights,
    model_select,
    source,
    show_result,
    save_result,
    output_path
    ):

    # Directory
    imgs_path = sorted(glob.glob(str(source) + '/*'))
    os.makedirs(f"{output_path}/pred", exist_ok=True)
    os.makedirs(f"{output_path}/bev", exist_ok=True)
    # calib = str(calib_file)

    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()

    # load weight
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    model = YOLO('yolo11m.pt')

    f_x, f_y, c_x, c_y = 1.113398681640625000e+03, 1.113261718750000000e+03, 4.881284790039062500e+02, 7.191560058593750000e+02
    calib = np.array([
        [f_x, 0, c_x, 0],
        [0, f_y, c_y, 0],
        [0, 0, 1, 0]
    ])

    plt.figure(figsize=(10, 10)) # for BEV
    tracked_stats = []

    # loop images
    for i, img_path in enumerate(imgs_path):

        img_pil = PIL.Image.open(img_path)

        dets2_ul = model.track(source=img_pil, classes=[0, 2, 3, 5], imgsz=640, device=0, conf=0.3, persist=True)[0]
        # dets2_ul = model.detect(img_pil, classes=[0, 2, 3, 5], imgsz=640, device=0, conf=0.3)[0]
        dets2 = []
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        for box in dets2_ul.boxes:
            cls_name = dets2_ul.names[box.cls.item()]
            bbox_list = box.xyxy[0].to(torch.int32).reshape(2,2).tolist()
            dets2.append(Bbox(bbox_list, cls_name, track_id=box.id))

        plt.clf()
        plt.xlim(-30, 30)
        plt.ylim(0, 60)

        for det in dets2:
            if not averages.recognized_class(det.detected_class):
                continue
            try: 
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib)
            except Exception as e:
                print("Error in detectedObject", e)
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = det.box_2d
            detected_class = det.detected_class

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            # predict orient, conf, and dim
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            # plot 3d detection
            track_id = str(int(det.track_id.item()))
            location = plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray, track_id=track_id)

            tracked_stats.append({
                "frame": i,
                "track_id": track_id,
                "class": det.detected_class,
                "location": location,
                "dim": dim.tolist(),
                "alpha": float(alpha)
            })

        if show_result:
           cv2.imshow('3d detection', img)
           cv2.waitKey(0)

        if save_result and output_path is not None:
            cv2.imwrite(f'{output_path}/pred/{i:03d}.png', img)
            plt.savefig(f'{output_path}/bev/{i:03d}.png')

    # Save stats to JSON
    with open(f"{output_path}/tracked_stats.json", "w") as f:
        json.dump(tracked_stats, f, indent=2)


def plot3d(
    img,
    proj_matrix,
    box_2d,
    dimensions,
    alpha,
    theta_ray,
    img_2d=None,
    track_id="-1",
    ):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    dx = 2 * np.cos(-orient)# + np.pi)
    dy = 2 * np.sin(-orient)# + np.pi)
    plt.arrow(location[0], location[2], dx, dy, head_width=1, head_length=1.5, fc='r', ec='r')
    plt.text(location[0] + 0.5, location[2]+ 0.5, track_id, color='red', fontsize=10)

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, proj_matrix, orient, dimensions, location) # 3d boxes

    return location

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'eval/image_2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=[0, 2, 3, 5], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default='weights/epoch_10.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=ROOT / 'output', help='Save output pat')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main(opt):
    detect3d(
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)