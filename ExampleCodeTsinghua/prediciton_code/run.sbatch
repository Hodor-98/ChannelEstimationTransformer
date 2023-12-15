#!/bin/bash
#SBATCH --job-name=EvalTransform
#SBATCH --output=EvalTransform%j.txt
#SBATCH --partition=elec.gpu.q
#SBATCH --gres=gpu:2

module load cuda11.1/toolkit/11.1.1 cuda11.1/profiler/11.1.1 cuda11.1/nsight/11.1.1 cuda11.1/fft/11.1.1 cuda11.1/blas/11.1.1 cudnn/8.1.1

python3 test_transformer_lstm_AR.py