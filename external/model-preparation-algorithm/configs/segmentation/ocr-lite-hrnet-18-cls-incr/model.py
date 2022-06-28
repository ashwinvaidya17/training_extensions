_base_ = [
  '../../../submodule/samples/cfgs/models/backbones/ote_litehrnet18.yaml',
  '../../../submodule/recipes/stages/_base_/models/segmentors/seg_class_incr.py'
]
# _base_ = [
#   '../../../submodule/models/segmentation/ocr_litehrnet18.yaml',
# ]

load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnet18_imagenet1k_rsc.pth'
