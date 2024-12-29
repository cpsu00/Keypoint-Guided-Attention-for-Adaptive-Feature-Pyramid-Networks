1. Intsall mmdetection (https://github.com/open-mmlab/mmdetection)
2. Build OpenCV for GPU and Non-free module
3. Overwrite the fpn.py file under mmdetection/mmdet/models/necks/fpn.py
4. Run python tools/train.py configs/mask_rcnn/mask-rcnn_r50_fpn_amp-1x_cityscapes.py
