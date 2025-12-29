#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_path> <vis_folder>"
    exit 1
fi

dataset_path=$1
vis_folder=$2

scenes=("02b33dfb-be2b-2d54-92d2-cd012b2b3c40" "fcf66d9e-622d-291c-84c2-bb23dfe31327")
# for scene in "${scenes[@]}"; do
#     python3 visualize2D.py --scene $scene --dataset_path $dataset_path --vis_folder $vis_folder &
# done

# for scene in "${scenes[@]}"; do
#     python3 visualize3D.py --scene $scene --dataset_path $dataset_path --vis_folder $vis_folder &
# done
# wait

# for scene in "${scenes[@]}"; do
#     python3 visualize3D_texts.py --scene $scene --dataset_path $dataset_path --vis_folder $vis_folder &
# done
# wait

for scene in "${scenes[@]}"; do
    # FIX: Added 'pad=ceil(iw/2)*2:ceil(ih/2)*2' to ensure dimensions are even
    ffmpeg -framerate 15 -i "$vis_folder/2D/$scene/frame-%06d.color.png" \
           -framerate 15 -i "$vis_folder/3D_text/$scene/frame-%06d.color.png" \
           -filter_complex "[1:v][0:v]scale2ref=iw:iw*ih/iw[rb][ra];[ra][rb]vstack=inputs=2,pad=ceil(iw/2)*2:ceil(ih/2)*2[v]" \
           -map "[v]" \
           -c:v libx264 -pix_fmt yuv420p -crf 24 -preset veryfast -shortest "${vis_folder}/${scene}.mp4"
done
wait