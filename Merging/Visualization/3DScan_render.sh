#!/bin/bash
set -euo pipefail

dataset_path=$1
vis_folder=$2
dataset_name="3RScan"

# Automatically find all scene IDs in the visualization folder
scenes=$(ls "$vis_folder/$dataset_name")

for scene in $scenes; do
    echo "Processing scene: $scene"
    python3 visualize2D.py --scene $scene --dataset_path $dataset_path --vis_folder $vis_folder &
done
wait

for scene in $scenes; do
    python3 visualize3D.py --scene $scene --dataset_path $dataset_path --vis_folder $vis_folder &
done
wait

for scene in $scenes; do
    python3 visualize3D_texts.py --scene $scene --dataset_path $dataset_path --vis_folder $vis_folder &
done
wait

for scene in $scenes; do
    echo "Rendering video for $scene..."
    ffmpeg -y -framerate 15 -i "$vis_folder/2D/$dataset_name/$scene/frame-%06d.color.png" \
           -framerate 15 -i "$vis_folder/3D_text/$dataset_name/$scene/frame-%06d.color.png" \
           -filter_complex "[1:v][0:v]scale2ref=iw:iw*ih/iw[rb][ra];[ra][rb]vstack=inputs=2[v]" \
           -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 24 -preset veryfast \
           "$vis_folder/${scene}.mp4"
done