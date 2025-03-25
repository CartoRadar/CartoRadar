#!/bin/bash

config_strings=("./config/building2-exp000/RadarOccNerf_offline.yaml" "./config/building2-exp001/RadarOccNerf_offline.yaml" "./config/building2-exp002/RadarOccNerf_offline.yaml"
"./config/building1-exp000/RadarOccNerf_offline.yaml" "./config/building1-exp001/RadarOccNerf_offline.yaml"
"./config/building3-exp000/RadarOccNerf_offline.yaml" "./config/building3-exp001/RadarOccNerf_offline.yaml" "./config/building3-exp002/RadarOccNerf_offline.yaml" "./config/building3-exp003/RadarOccNerf_offline.yaml"
"./config/building4-exp000/RadarOccNerf_offline.yaml" "./config/building4-exp001/RadarOccNerf_offline.yaml" "./config/building4-exp002/RadarOccNerf_offline.yaml"
"./config/building5-exp000/RadarOccNerf_offline.yaml" "./config/building5-exp001/RadarOccNerf_offline.yaml")

expnames=("building2_exp000_offline_default" "building2_exp001_offline_default" "building2_exp002_offline_default"
"building1-exp000_offline_default" "building1-exp001_offline_default"
"building3-exp000_offline_default" "building3-exp001_offline_default" "building3-exp002_offline_default" "building3-exp003_offline_default"
"building4-exp000_offline_default" "building4-exp001_offline_default" "building4-exp002_offline_default"
"building5-exp000_offline_default" "building5-exp001_offline_default")

if [ ${#config_strings[@]} -ne ${#expnames[@]} ]; then
  echo "Error: The number of config strings and experiment names must be equal."
  exit 1
fi

for i in "${!config_strings[@]}"; do
  config_string=${config_strings[$i]}
  expname="${expnames[$i]}"

  current_time=$(date +"_%Y%m%d_%H%M%S")
  expname="${expname}${current_time}"
  echo "exp name: $expname"

  echo "Training"
  python ./main_offline.py --config "$config_string" --expname "$expname"

  echo "Generating pkl"
  python ./mobicom_generate_pkl.py --target "$expname" --save_pkl

  echo "Evaluating"
  python ./mobicom_analyze_pkl.py --target "$expname"
done

python result_summary.py