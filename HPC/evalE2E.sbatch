#!/bin/bash
#SBATCH --job-name=EvalE2E
#SBATCH --output=log/EvalE2E_output_%j.txt
#SBATCH --partition=elec.gpu.q
#SBATCH --gres=gpu:1

module load cuda11.5/toolkit/11.5.1 cuda11.5/profiler/11.5.1 cuda11.5/nsight/11.5.1 cuda11.5/fft/11.5.1 cuda11.5/blas/11.5.1 cudnn/8.1.1

conda run -n sionna python3 python/EvaluateE2E.py