#!/bin/bash
#SBATCH --job-name=run_bf_cuda
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mihail.stamenov@studio.unibo.it
#SBATCH --output=output.txt
#SBATCH --error=errors.txt
#SBATCH --time=06:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

echo "Compiling soutce solution..."
make source
./bf_source