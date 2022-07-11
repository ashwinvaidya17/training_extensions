_base_ = [
  '../../../submodule/models/segmentation/ocr_litehrnet18_mod2.yaml',
]

load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnet18_imagenet1k_rsc.pth'
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()