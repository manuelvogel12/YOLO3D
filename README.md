# YOLO For 3D Object Detection

### This Fork has the following changes frinn the [original Repo](https://github.com/ruhyadi/YOLO3D):
* newer python, etc
* YOLO 11
* Object Tracking
* More Visualization
* ...

ORIGINAL README:


#### Note
I have created a new repository of improvements of YOLO3D wrapped in pytorch lightning and more various object detector backbones, currently on development. Please check [ruhyadi/yolo3d-lightning](https://github.com/ruhyadi/yolo3d-lightning).

Unofficial implementation of [Mousavian et al](https://arxiv.org/abs/1612.00496) in their paper *3D Bounding Box Estimation Using Deep Learning and Geometry*. YOLO3D uses a different approach, as the detector uses YOLOv5 which previously used Faster-RCNN, and Regressor uses ResNet18/VGG11 which was previously VGG19.

![inference](docs/demo.gif)

## Installation
For installation you can use virtual environment like conda, e.g.:

### Conda Virtual Env
Create conda environment
```
conda create -n yolo3d python=3.12
```
Install PyTorch and torchvision. If your GPU doesn't support it, please follow [Nelson Liu blogs](https://github.com/nelson-liu/pytorch-manylinux-binaries). 
```
pip install torch torcvision
```
Last, install requirements
```
pip install -r requirements.txt
```


### Download Pretrained Weights
In order to run inference code or resuming training, you can download pretrained ResNet18 or VGG11 model. I have train model with 10 epoch each. You can download model with `resnet18` or `vgg11` for `--weights` arguments.
```
cd ${YOLO3D_DIR}/weights
python get_weights.py --weights resnet18
```

## Inference
For inference with pretrained model you can run code below. It can be run in conda env or docker container. 
```
python inference.py \
    --source eval/image_2 \
    --reg_weights weights/resnet18.pkl \
    --model_select resnet18 \
    --show_result --save_result
```

## Training (NOT ADJUSTED YET)
YOLO3D model can be train with PyTorch or PyTorch Lightning. In order to train you need API KEY for [comet.ml](https://www.comet.ml) (visualize your training loss/accuracy). Follow comet.ml documentation to get API key.
```
python train.py \
    --epochs 10 \
    --batch_size 32 \
    --num_workers 2 \
    --save_epoch 5 \
    --train_path ./dataset/KITTI/training \
    --model_path ./weights \
    --select_model resnet18 \
    --api_key xxx
```
In order train with pytorch lightning run code below:
```
!python train_lightning.py \
    --train_path dataset/KITTI/training \
    --checkpoint_path weights/checkpoints \
    --model_select resnet18 \
    --epochs 10 \
    --batch_size 32 \
    --num_workers 2 \
    --gpu 1 \
    --val_split 0.1 \
    --model_path weights \
    --api_key xxx
```

## Reference
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [shakdem/3D-BoungingBox](https://github.com/skhadem/3D-BoundingBox)

```
@misc{mousavian20173d,
      title={3D Bounding Box Estimation Using Deep Learning and Geometry}, 
      author={Arsalan Mousavian and Dragomir Anguelov and John Flynn and Jana Kosecka},
      year={2017},
      eprint={1612.00496},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
