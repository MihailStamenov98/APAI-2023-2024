#!/ bin / bash
#SBATCH-- job - name = compile_and_run_solutions
#SBATCH-- output = output_ % j.txt
#SBATCH-- error = errors_ % j.txt
#SBATCH-- time = 01 : 00 : 00
#SBATCH-- mail - user = mihail.stamenov @studio.unibo.it
#SBATCH-- partition = normal
#SBATCH-- ntasks = 1
#SBATCH-- cpus - per - task = 4
#SBATCH-- gpus = 1
#SBATCH-- mem = 8G

#Load necessary modules if any(e.g., CUDA)
module load gcc
module load cuda

#Navigate to the solvers directory
cd src/solvers/

#Compile and execute the first solution
echo "Compiling dest solution..."
make dest
if [ $? -eq 0 ]; then
    echo "Running bf_dest..."
    ./bf_dest
else
    echo "Failed to compile bf_dest."
    exit 1
fi

#Compile and execute the second solution
echo "Compiling source solution..."
make source
if [ $? -eq 0 ]; then
    echo "Running bf_source..."
    ./bf_source
else
    echo "Failed to compile bf_source."
    exit 1
fi

#Compile and execute the CUDA solution
echo "Compiling CUDA solution..."
nvcc -o bf_cuda ./bellman_ford_cuda.cu
if [ $? -eq 0 ];
then echo "Running bf_cuda..."./ bf_cuda else echo
    "Failed to compile bf_cuda." exit 1 fi

        echo "All tasks completed successfully."
