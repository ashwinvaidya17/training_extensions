#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
repeats=(0 1 2)
seeds=(0 1 2)
num_imgs=30
channels=1024
# Multilabel, 30 images
for seed in ${seeds[@]}
    do
        for repeat in ${repeats[@]}
        do
        start_time=`date +%y/%m/%d-%T`
        dirname=data/mlc_seg_exp_final/pascal_voc_${num_imgs}/mlc_segmentor_1024/
        mkdir -p ${dirname}
        logname=${dirname}/training_log_seed_${seed}_repeat_${repeat}.txt
        echo exp_start:$start_time > ${logname}
        model_template=./external/mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2-${channels}/template_experimental.yaml
        ote train ${model_template} \
        --train-ann-files data/pascal_voc_shortened_${num_imgs}_seed_${seed}/annotations/training \
        --train-data-roots data/pascal_voc_shortened_${num_imgs}_seed_${seed}/images/training \
        --val-ann-files data/pascal_voc_shortened_${num_imgs}_seed_${seed}/annotations/validation \
        --val-data-roots data/pascal_voc_shortened_${num_imgs}_seed_${seed}/images/validation \
        --save-model-to ${dirname} 2>&1 | tee -a ${logname}
        end_time=`date +%y/%m/%d-%T`
        echo exp_end:$end_time >> ${logname}
        done
    done

num_imgs=60
# Multilabel, 60 images
for seed in ${seeds[@]}
    do
        for repeat in ${repeats[@]}
        do
        start_time=`date +%y/%m/%d-%T`
        dirname=data/mlc_seg_exp_final/pascal_voc_${num_imgs}/mlc_segmentor_1024/
        mkdir -p ${dirname}
        logname=${dirname}/training_log_seed_${seed}_repeat_${repeat}.txt
        echo exp_start:$start_time > ${logname}
        model_template=./external/mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2-${channels}/template_experimental.yaml
        ote train ${model_template} \
        --train-ann-files data/pascal_voc_shortened_${num_imgs}_seed_${seed}/annotations/training \
        --train-data-roots data/pascal_voc_shortened_${num_imgs}_seed_${seed}/images/training \
        --val-ann-files data/pascal_voc_shortened_${num_imgs}_seed_${seed}/annotations/validation \
        --val-data-roots data/pascal_voc_shortened_${num_imgs}_seed_${seed}/images/validation \
        --save-model-to ${dirname} 2>&1 | tee -a ${logname}
        end_time=`date +%y/%m/%d-%T`
        echo exp_end:$end_time >> ${logname}
        done
    done

num_imgs=120
# Multilabel, 120 images
for seed in ${seeds[@]}
    do
        for repeat in ${repeats[@]}
        do
        start_time=`date +%y/%m/%d-%T`
        dirname=data/mlc_seg_exp_final/pascal_voc_${num_imgs}/mlc_segmentor_1024/
        mkdir -p ${dirname}
        logname=${dirname}/training_log_seed_${seed}_repeat_${repeat}.txt
        echo exp_start:$start_time > ${logname}
        model_template=./external/mmsegmentation/configs/custom-sematic-segmentation/ocr-lite-hrnet-18-mod2-${channels}/template_experimental.yaml
        ote train ${model_template} \
        --train-ann-files data/pascal_voc_shortened_${num_imgs}_seed_${seed}/annotations/training \
        --train-data-roots data/pascal_voc_shortened_${num_imgs}_seed_${seed}/images/training \
        --val-ann-files data/pascal_voc_shortened_${num_imgs}_seed_${seed}/annotations/validation \
        --val-data-roots data/pascal_voc_shortened_${num_imgs}_seed_${seed}/images/validation \
        --save-model-to ${dirname} 2>&1 | tee -a ${logname}
        end_time=`date +%y/%m/%d-%T`
        echo exp_end:$end_time >> ${logname}
        done
    done