#!/bin/bash

model_name=("mobilenetv2_atss_cls_incr" "resnet50_vfnet_cls_incr")
output_path="../ote_poc_data/mixed_precision/det"


git checkout soobee/mixed_precision
mode="fp16"

for model in "${model_name[@]}"
do
    
    #missing 50
    data="car_bicycle_bus_train"
    
    mkdir -p $output_path/${model}/${data}/$mode

    cmd="ote train \
    external/model-preparation-algorithm/configs/detection/$model/template.yaml \
    --train-ann-files ../ote_poc_data/det/coco-annotation/$data/instances_train2017.1234@1.0_OLD_NEW_12_MR50.json \
    --train-data-roots ../ote_poc_data/det/coco/images/train2017 \
    --val-ann-files ../ote_poc_data/det/coco-annotation/$data/instances_val2017.1234@10.0_OLD_NEW.json \
    --val-data-roots ../ote_poc_data/det/coco/images/val2017 \
    --load-weights ../ote_poc_data/det/pretrained/mr50/initial_model/weights.pth \
    --save-model-to $output_path/${model}/${data}/$mode params --learning_parameters.num_iters 1"
    echo $cmd; CUDA_VISIBLE_DEVICES=0 $cmd &> $output_path/${model}/${data}/$mode/train.log

    cmd="ote eval \
    external/model-preparation-algorithm/configs/detection/$model/template.yaml \
    --test-ann-files ../ote_poc_data/det/coco-annotation/$data/instances_test2017.1234@100.0_OLD_NEW.json \
    --test-data-roots ../ote_poc_data/det/coco/images/val2017 \
    --load-weights  $output_path/${model}/${data}/$mode/weights.pth"
    echo $cmd; CUDA_VISIBLE_DEVICES=0 $cmd &> $output_path/${model}/${data}/test.log

    #lvis
    data="LVIS-subsets"
    mkdir -p $output_path/${model}/${data}

    cmd="ote train \
    external/model-preparation-algorithm/configs/detection/$model/template.yaml \
    --train-ann-files ../ote_poc_data/det/coco-annotation/$data/lvis_train_seed1234@1.0_OLD_NEW_1CLS.json \
    --train-data-roots ../ote_poc_data/det/coco/images \
    --val-ann-files ../ote_poc_data/det/coco-annotation/$data/lvis_val_seed1234@10.0_OLD_NEW_1CLS.json \
    --val-data-roots ../ote_poc_data/det/coco/images \
    --load-weights ../ote_poc_data/det/pretrained/initial_model_hpo/weights.pth \
    --save-model-to $output_path/${model}/${data}/$mode params --learning_parameters.num_iters 1"
    echo $cmd; CUDA_VISIBLE_DEVICES=0 $cmd &> $output_path/${model}/${data}/$mode/test.log

    cmd="ote eval \
    external/model-preparation-algorithm/configs/detection/$model/template.yaml \
    --test-ann-files ../ote_poc_data/det/coco-annotation/$data/lvis_test_seed1234@100.0_OLD_NEW_1CLS.json \
    --test-data-roots ../ote_poc_data/det/coco/images \
    --load-weights  $output_path/${model}/${data}/$mode/weights.pth"
    echo $cmd; CUDA_VISIBLE_DEVICES=0 $cmd &> $output_path/${model}/${data}/$mode/test.log
done


git checkout sc-intg-ins-seg
mode="fp32"

for model in "${model_name[@]}"
do
    
    #missing 50
    data="car_bicycle_bus_train"
    
    mkdir -p $output_path/${model}/${data}/$mode

    cmd="ote train \
    external/model-preparation-algorithm/configs/detection/$model/template.yaml \
    --train-ann-files ../ote_poc_data/det/coco-annotation/$data/instances_train2017.1234@1.0_OLD_NEW_12_MR50.json \
    --train-data-roots ../ote_poc_data/det/coco/images/train2017 \
    --val-ann-files ../ote_poc_data/det/coco-annotation/$data/instances_val2017.1234@10.0_OLD_NEW.json \
    --val-data-roots ../ote_poc_data/det/coco/images/val2017 \
    --load-weights ../ote_poc_data/det/pretrained/mr50/initial_model/weights.pth \
    --save-model-to $output_path/${model}/${data}/$mode params --learning_parameters.num_iters 1"
    echo $cmd; CUDA_VISIBLE_DEVICES=0 $cmd &> $output_path/${model}/${data}/$mode/train.log

    cmd="ote eval \
    external/model-preparation-algorithm/configs/detection/$model/template.yaml \
    --test-ann-files ../ote_poc_data/det/coco-annotation/$data/instances_test2017.1234@100.0_OLD_NEW.json \
    --test-data-roots ../ote_poc_data/det/coco/images/val2017 \
    --load-weights  $output_path/${model}/${data}/$mode/weights.pth"
    echo $cmd; CUDA_VISIBLE_DEVICES=0 $cmd &> $output_path/${model}/${data}/test.log

    #lvis
    data="LVIS-subsets"
    mkdir -p $output_path/${model}/${data}

    cmd="ote train \
    external/model-preparation-algorithm/configs/detection/$model/template.yaml \
    --train-ann-files ../ote_poc_data/det/coco-annotation/$data/lvis_train_seed1234@1.0_OLD_NEW_1CLS.json \
    --train-data-roots ../ote_poc_data/det/coco/images \
    --val-ann-files ../ote_poc_data/det/coco-annotation/$data/lvis_val_seed1234@10.0_OLD_NEW_1CLS.json \
    --val-data-roots ../ote_poc_data/det/coco/images \
    --load-weights ../ote_poc_data/det/pretrained/initial_model_hpo/weights.pth \
    --save-model-to $output_path/${model}/${data}/$mode params --learning_parameters.num_iters 1"
    echo $cmd; CUDA_VISIBLE_DEVICES=0 $cmd &> $output_path/${model}/${data}/$mode/test.log

    cmd="ote eval \
    external/model-preparation-algorithm/configs/detection/$model/template.yaml \
    --test-ann-files ../ote_poc_data/det/coco-annotation/$data/lvis_test_seed1234@100.0_OLD_NEW_1CLS.json \
    --test-data-roots ../ote_poc_data/det/coco/images \
    --load-weights  $output_path/${model}/${data}/$mode/weights.pth"
    echo $cmd; CUDA_VISIBLE_DEVICES=0 $cmd &> $output_path/${model}/${data}/$mode/test.log
done