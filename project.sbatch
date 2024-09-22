#!/bin/bash
#SBATCH --job-name=compile_and_run_solutions
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mihail.stamenov@studio.unibo.it
#SBATCH --output=output.txt
#SBATCH --error=errors.txt
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1


echo "Generate graphs..."
cd src/generate_graphs
make
if [ $? -eq 0 ]; then
    echo "Creating graphs..."
    ./generate_graphs
else
    echo "Failed to compile bf_dest."
    exit 1
fi


cd ../solvers/


echo "Compiling source solution..."
make source
if [ $? -eq 0 ]; then
    echo "Running bf_source..."
    ./bf_source
else
    echo "Failed to compile bf_source."
    exit 1
fi

echo "Compiling CUDA solution..."
nvcc -o bf_cuda ./bellman_ford_cuda.cu
if [ $? -eq 0 ];then 
    echo "Running bf_cuda..."
    ./bf_cuda 
else 
    echo "Failed to compile bf_cuda." 
    exit 1 
fi

echo "Compiling dest solution..."
make dest
if [ $? -eq 0 ]; then
    echo "Running bf_dest..."
    ./bf_dest
else
    echo "Failed to compile bf_dest."
    exit 1
fi


echo "Compiling calculations..."
make calculate
if [ $? -eq 0 ]; then
    echo "Running calculations..."
    ./calculate_results
else
    echo "Failed to compile calculations."
    exit 1
fi

wait

echo "All tasks completed successfully."
