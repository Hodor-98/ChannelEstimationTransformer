#!/bin/bash
#SBATCH --job-name=GenerateChannels
#SBATCH --output=log/GenerateChannels_output_%j.txt
#SBATCH --partition=elec.gpu-es02.q
#SBATCH --gres=gpu:1

module load cuda11.5/toolkit/11.5.1 cuda11.5/profiler/11.5.1 cuda11.5/nsight/11.5.1 cuda11.5/fft/11.5.1 cuda11.5/blas/11.5.1 cudnn/8.1.1


# pip3 install torch==1.8.1


time python3 python/GenerateChannels.py