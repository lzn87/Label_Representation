#!/bin/bash
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=iwilds
#SBATCH --output=iwilds_wildcam_category_resnet110.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:8

nvidia-smi
python check_cuda.py
python train_custom.py --model resnet110 \
					--dataset iwildcam \
					--label category \
					--seed 100 \
					--data_dir /work/zli/wilds \
					--base_dir /work/zli/ \
					--batch_size 1024