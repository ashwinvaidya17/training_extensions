#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# MULTILABEL-1024, FULL
start_time=`date +%y/%m/%d-%T`
dirname=data/mlc_seg_exp_final/pascal_voc_full/mlc_segmentor_1024/
mkdir -p ${dirname}
logname=${dirname}/training_log.txt
echo exp_start:$start_time > ${logname}
model_template=./external/mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2-1024/template_experimental.yaml
ote train ${model_template} \
--train-ann-files data/pascal_voc/annotations/training \
--train-data-roots data/pascal_voc/images/training \
--val-ann-files data/pascal_voc/annotations/validation \
--val-data-roots data/pascal_voc/images/validation \
--save-model-to ${dirname} 2>&1 | tee -a ${logname}
end_time=`date +%y/%m/%d-%T`
echo exp_end:$end_time >> ${logname}

# MULTILABEL-2048, FULL
start_time=`date +%y/%m/%d-%T`
dirname=data/mlc_seg_exp_final/pascal_voc_full/mlc_segmentor_2048/
mkdir -p ${dirname}
logname=${dirname}/training_log.txt
echo exp_start:$start_time > ${logname}
model_template=./external/mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2-2048/template_experimental.yaml
ote train ${model_template} \
--train-ann-files data/pascal_voc/annotations/training \
--train-data-roots data/pascal_voc/images/training \
--val-ann-files data/pascal_voc/annotations/validation \
--val-data-roots data/pascal_voc/images/validation \
--save-model-to ${dirname} 2>&1 | tee -a ${logname}
end_time=`date +%y/%m/%d-%T`
echo exp_end:$end_time >> ${logname}

# ORIGINAL, FULL
start_time=`date +%y/%m/%d-%T`
dirname=data/mlc_seg_exp_final/pascal_voc_full/origin_segmentor/
mkdir -p ${dirname}
logname=${dirname}/training_log.txt
echo exp_start:$start_time > ${logname}
model_template=./external/mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2_origin/template_experimental.yaml
ote train ${model_template} \
--train-ann-files data/pascal_voc/annotations/training \
--train-data-roots data/pascal_voc/images/training \
--val-ann-files data/pascal_voc/annotations/validation \
--val-data-roots data/pascal_voc/images/validation \
--save-model-to ${dirname} 2>&1 | tee -a ${logname}
end_time=`date +%y/%m/%d-%T`
echo exp_end:$end_time >> ${logname}