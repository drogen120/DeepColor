#!/usr/bin/env sh

TRAIN_DIR=./logs_colornet/
python convert_model2lut.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=lisa \
    --dataset_split_name=train \
    --model_name=colornet \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=1
