# _base_ = [
#  '../../../submodule/samples/cfgs/models/backbones/ote_litehrnet_s_mod2.yaml',
#  '../../../submodule/recipes/stages/_base_/models/segmentors/seg_class_incr_small.py'
# ]
_base_ = [
  '../../../submodule/models/segmentation/ocr_litehrnet_s_mod2.yaml',
]

load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnet18_imagenet1k_rsc.pth'
