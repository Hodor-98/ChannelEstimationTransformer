#!/bin/bash
#SBATCH --job-name=TrainTransformer
#SBATCH --output=log/TrainTransformer_output_%j.txt
#SBATCH --partition=elec.gpu.q
#SBATCH --gres=gpu:4


python3 python/Train.py

