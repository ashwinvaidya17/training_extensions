#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# ORIGNAL
# start_time=`date +%y/%m/%d-%T`
# dirname=data/mlc_seg_pascal_voc_2007_full/mlc_segmentor_0.5_asl_0.1_loss/
# mkdir -p ${dirname}
# logname=${dirname}/training_log.txt
# echo exp_start:$start_time > ${logname}
# model_template=./external/mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2/template_experimental.yaml
# ote train ${model_template} \
# --train-ann-files data/pascal_voc/annotations/training \
# --train-data-roots data/pascal_voc/images/training \
# --val-ann-files data/pascal_voc/annotations/validation \
# --val-data-roots data/pascal_voc/images/validation \
# --save-model-to ${dirname} 2>&1 | tee -a ${logname}
# end_time=`date +%y/%m/%d-%T`
# echo exp_end:$end_time >> ${logname}


start_time=`date +%y/%m/%d-%T`
dirname=data/mlc_seg_pascal_voc_2007_short/mlc_segmentor_2048_asl_0_0/
mkdir -p ${dirname}
logname=${dirname}/training_log.txt
echo exp_start:$start_time > ${logname}
model_template=./external/mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2/template_experimental.yaml
ote train ${model_template} \
--train-ann-files data/pascal_voc_shortened/annotations/training \
--train-data-roots data/pascal_voc_shortened/images/training \
--val-ann-files data/pascal_voc_shortened/annotations/validation \
--val-data-roots data/pascal_voc_shortened/images/validation \
--save-model-to ${dirname} 2>&1 | tee -a ${logname}
end_time=`date +%y/%m/%d-%T`
echo exp_end:$end_time >> ${logname}

# start_time=`date +%y/%m/%d-%T`
# dirname=data/mlc_seg_pascal_voc_2007_short_120_120/mlc_segmentor_0.5/
# mkdir -p ${dirname}
# logname=${dirname}/training_log.txt
# echo exp_start:$start_time > ${logname}
# model_template=./external/mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2/template_experimental.yaml
# ote train ${model_template} \
# --train-ann-files data/pascal_voc_shortened/annotations/training \
# --train-data-roots data/pascal_voc_shortened/images/training \
# --val-ann-files data/pascal_voc_shortened/annotations/validation \
# --val-data-roots data/pascal_voc_shortened/images/validation \
# --save-model-to ${dirname} 2>&1 | tee -a ${logname}
# end_time=`date +%y/%m/%d-%T`
# echo exp_end:$end_time >> ${logname}