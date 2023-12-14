#!/bin/bash
#SBATCH --job-name=TrainTransformer
#SBATCH --output=TrainTransformer_output_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuDNN/8.7.0.84-CUDA-11.8.0 

conda run -n training python3 Train.py

