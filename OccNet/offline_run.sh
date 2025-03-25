#!/bin/bash

config_string=$1
expname=$2

current_time=$(date +"_%Y%m%d_%H%M%S")

expname="${expname}${current_time}"
echo "exp name: $expname"

echo "Training"
python ./main_offline.py --config "$config_string" --expname "$expname"

echo "Generating pkl"
python ./mobicom_generate_pkl.py --target "$expname" --save_pkl

echo "Evaluating"
python ./mobicom_analyze_pkl.py --target "$expname"