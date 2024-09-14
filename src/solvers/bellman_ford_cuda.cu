
/*
 * This is a CUDA version of bellman_ford algorithm
 * Compile: nvcc -arch=sm_52 -o cuda_bellman_ford cuda_bellman_ford.cu
 * Run: ./cuda_bellman_ford <input file> <number of blocks per grid> <number of
 * threads per block>, you will find the output file 'output.txt'
 * */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INF 1000000

typedef struct
{
    bool hasNegativeCycle;
    int negativeCycleNode;
    double timeInSeconds;
    int numberNodes;
    int edgesCount;
    int startNode;
    int *predecessor;
    int *dist;

} BFOutput;
void freeBFOutput(BFOutput *output)
{
    if (output != NULL)
    {
        if (output->predecessor != NULL)
        {
            free(output->predecessor);
        }
        if (output->dist != NULL)
        {
            free(output->dist);
        }
        free(output);
    }
}

void readSourceGraphFromFileToDevice(const char *filename,
                                     int **&d_neighbouringNodes,
                                     int **&d_neighbouringNodesWeights, int *&n,
                                     int *&edgesCount, int *&neighboursCount)
{
    int **neighbouringNodes;
    int **neighbouringNodesWeights;
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }
    n = (int *)malloc(sizeof(int));
    edgesCount = (int *)malloc(sizeof(int));

    fscanf(file, "g %d %d\n", n, edgesCount);
    neighbouringNodes = (int **)malloc(*n * sizeof(int *));
    neighbouringNodesWeights = (int **)malloc(*n * sizeof(int *));
    d_neighbouringNodes = (int **)malloc(*n * sizeof(int *));
    d_neighbouringNodesWeights = (int **)malloc(*n * sizeof(int *));
    neighboursCount = (int *)malloc(*n * sizeof(int));
    int *indexeForNode = (int *)malloc(*n * sizeof(int));
    int numEdgesOut = 0;
    for (int i = 0; i < *n; i++)
    {
        int temp;
        fscanf(file, "n %d %d\n", &temp, &neighboursCount[i]);
        cudaMalloc(&d_neighbouringNodes[i], neighboursCount[i] * sizeof(int));
        cudaMalloc(&d_neighbouringNodesWeights[i],
                   neighboursCount[i] * sizeof(int));
        neighbouringNodes[i] = (int *)malloc(neighboursCount[i] * sizeof(int));
        neighbouringNodesWeights[i] =
            (int *)malloc(neighboursCount[i] * sizeof(int));
        indexeForNode[i] = 0;
        numEdgesOut = numEdgesOut + neighboursCount[i];
    }

    for (int i = 0; i < numEdgesOut; i++)
    {
        int source;
        int dest;
        int weight;
        fscanf(file, "e %d %d %d\n", &source, &dest, &weight);
        neighbouringNodes[source][indexeForNode[source]] = dest;
        neighbouringNodesWeights[source][indexeForNode[source]] = weight;
        indexeForNode[source]++;
    }
    for (int i = 0; i < *n; i++)
    {
        cudaMemcpy(d_neighbouringNodes[i], neighbouringNodes[i],
                   neighboursCount[i] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_neighbouringNodesWeights[i], neighbouringNodesWeights[i],
                   neighboursCount[i] * sizeof(int), cudaMemcpyHostToDevice);
    }

    fclose(file);
}
double gettime(void)
{
    /*#ifdef _WIN32
        LARGE_INTEGER frequency;
        LARGE_INTEGER start;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start);
        return (double)start.QuadPart / frequency.QuadPart;
    #else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);*/
    return 5.0;
    // #endif
}

__global__ void relax_initial(int *d_dist, int *d_predecessor,
                              bool *d_hasChanged, int n, int startNode,
                              int maxVal)
{
    int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
    int i = bdim * bid + tid;
    int skip = bdim * gdim;
    for (int k = i; k < n; k += skip)
    {
        d_predecessor[k] = -1;
        d_hasChanged[k] = false;

        if (k != startNode)
        {
            d_dist[k] = maxVal;
        }
        else
        {
            d_dist[startNode] = 0;
        }
    }
    __syncthreads();
}
__global__ void copyHasChanged(bool *hasChanged, int n)
{
    int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
    int i = bdim * bid + tid;
    int skip = bdim * gdim;

    for (int j = i; j < n; j += skip)
    {
        hasChanged[j] = false;
    }
    __syncthreads();
}

__global__ void bellmanFordIteration(int *weights, int *neighbours,
                                     int neighboursCount, int *predecessor,
                                     int *dist, bool *hasChanged, int source)
{
    int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
    int i = bdim * bid + tid;
    int skip = bdim * gdim;
    for (int edgeIndex = i; edgeIndex < neighboursCount; edgeIndex += skip)
    {

        int destination = neighbours[edgeIndex];
        int weight = weights[edgeIndex];
        int new_dist = dist[source] + weight;
        if (new_dist < dist[destination])
        {
            hasChanged[destination] = true;
            dist[destination] = new_dist;
            predecessor[destination] = source;
        }
    }
    __syncthreads();
}

// Kernel to perform block-level reduction
__global__ void blockReduceOr(bool *input, bool *blockResults, int size)
{
    extern __shared__ bool shared[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (index < size)
    {
        shared[tid] = input[index];
    }
    else
    {
        shared[tid] = false; // If index is out of bounds, use false
    }

    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] = shared[tid] || shared[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block's reduction to the blockResults array
    if (tid == 0)
    {
        blockResults[blockIdx.x] = shared[0];
    }
}

// Function to handle multi-pass reduction
bool reduceLargeArray(bool *d_input, int size, int blockSize)
{
    int gridSize = (size + blockSize - 1) / blockSize;
    bool *d_intermediate;
    bool h_result;

    // Allocate memory for intermediate results
    cudaMalloc((void **)&d_intermediate, gridSize * sizeof(bool));

    // Continue reducing the array in multiple passes until gridSize == 1
    while (gridSize > 1)
    {
        size_t sharedMemorySize = blockSize * sizeof(bool);

        // Launch kernel to reduce the current input into d_intermediate
        blockReduceOr<<<gridSize, blockSize, sharedMemorySize>>>(
            d_input, d_intermediate, size);
        cudaDeviceSynchronize();
        // Update size to reflect the size of the intermediate results
        size = gridSize;
        gridSize = (size + blockSize - 1) / blockSize;

        // Swap input and intermediate buffers (reuse input as the intermediate
        // result)
        bool *temp = d_input;
        d_input = d_intermediate;
        d_intermediate = temp;
    }

    // Now the gridSize is 1, reduce the final intermediate result to a single
    // value
    size_t sharedMemorySize = blockSize * sizeof(bool);
    blockReduceOr<<<1, blockSize, sharedMemorySize>>>(d_input, d_intermediate,
                                                      size);

    // Copy the final result back to the host
    cudaMemcpy(&h_result, d_intermediate, sizeof(bool), cudaMemcpyDeviceToHost);

    // Free intermediate memory
    cudaFree(d_intermediate);

    return h_result;
}

void freeMem(int *d_dist, int **d_neighbouringNodes,
             int **d_neighbouringNodesWeights, int *neighboursCount,
             int *d_predecessor, bool *d_hasChanged, int size)
{
    cudaFree(d_dist);
    cudaFree(d_hasChanged);
    cudaFree(d_predecessor);
    free(neighboursCount);
    for (int i = 0; i < size; ++i)
    {
        cudaFree(d_neighbouringNodes[i]);
        cudaFree(d_neighbouringNodesWeights[i]);
    }
    free(d_neighbouringNodes);
    free(d_neighbouringNodesWeights);
}

BFOutput *initBFOutput(int startNode, int size, int edgesCount)
{
    BFOutput *result;
    result = (BFOutput *)malloc(sizeof(BFOutput));
    result->startNode = startNode;
    result->predecessor = (int *)malloc(size * sizeof(int));
    result->dist = (int *)malloc(size * sizeof(int));
    result->negativeCycleNode = -1;
    result->numberNodes = size;
    result->edgesCount = edgesCount;
    return result;
}

BFOutput *bellmanFordCuda(const char *filename, int startNode)
{
    // Pointer to the graph on the device
    int **d_neighbouringNodes;
    int **d_neighbouringNodesWeights;
    int *n;
    int *edgesCount;
    int *neighboursCount;
    readSourceGraphFromFileToDevice(filename, d_neighbouringNodes,
                                    d_neighbouringNodesWeights, n, edgesCount,
                                    neighboursCount);
    int size = *n;
    int *h_dist = (int *)malloc(size * sizeof(int));
    int *d_dist;
    cudaMalloc(&d_dist, size * sizeof(int));

    int *h_predecessor = (int *)malloc(size * sizeof(int));
    int *d_predecessor;
    cudaMalloc(&d_predecessor, size * sizeof(int));

    bool *wasUpdatedLastIter = (bool *)malloc(size * sizeof(bool));

    bool *h_hasChanged = (bool *)malloc(size * sizeof(bool));
    bool *d_hasChanged;
    cudaMalloc(&d_hasChanged, size * sizeof(bool));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    // Call the function to copy the SourceGraph to the GPU

    BFOutput *result = initBFOutput(startNode, size, (*edgesCount));

    dim3 gdim(blocksPerGrid);
    dim3 bdim(threadsPerBlock);
    double tstart, tend;
    tstart = gettime();
    relax_initial<<<gdim, bdim>>>(d_dist, d_predecessor, d_hasChanged, size,
                                  startNode, INF);
    cudaMemcpy(wasUpdatedLastIter, d_hasChanged, size * sizeof(bool),
               cudaMemcpyDeviceToHost);
    wasUpdatedLastIter[startNode] = true;
    cudaDeviceSynchronize(); // wait for kernel to finish

    for (int iter = 0; iter < size; iter++)
    {

        for (int source = 0; source < (*n); ++source)
        {
            if (wasUpdatedLastIter[source])
            {
                bellmanFordIteration<<<gdim, bdim>>>(
                    d_neighbouringNodesWeights[source], d_neighbouringNodes[source],
                    neighboursCount[source], d_predecessor, d_dist, d_hasChanged,
                    source);
            }
            cudaDeviceSynchronize();
            bool hasChange = reduceLargeArray(d_hasChanged, size, threadsPerBlock);
            if (iter == size - 1 && hasChange)
            {
                tend = gettime();
                cudaMemcpy((*result).predecessor, d_predecessor, size * sizeof(int),
                           cudaMemcpyDeviceToHost);
                (*result).negativeCycleNode = source;
                (*result).hasNegativeCycle = true;
                (*result).timeInSeconds = tend - tstart;

                freeMem(d_dist, d_neighbouringNodes, d_neighbouringNodesWeights,
                        neighboursCount, d_predecessor, d_hasChanged, size);
                return result;
            }
        }
        cudaMemcpy(wasUpdatedLastIter, d_hasChanged, size * sizeof(bool),
                   cudaMemcpyDeviceToHost);
        copyHasChanged<<<gdim, bdim>>>(d_hasChanged, size);
        cudaDeviceSynchronize();
    }

    tend = gettime();
    cudaMemcpy((*result).dist, d_dist, size * sizeof(int),
               cudaMemcpyDeviceToHost);
    (*result).hasNegativeCycle = false;
    (*result).timeInSeconds = tend - tstart;

    freeMem(d_dist, d_neighbouringNodes, d_neighbouringNodesWeights,
            neighboursCount, d_predecessor, d_hasChanged, size);
    return result;
}

void writeResult(BFOutput *out, const char *filename, bool writeAll)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }
    if ((*out).hasNegativeCycle)
    {
        fprintf(file, "There is negative cycle in the graph\n");
        fprintf(file, "From this node one can bactrack to find the cycle = %d\n",
                (*out).negativeCycleNode);
    }
    else
    {
        fprintf(file, "There is NOT negative cycle in the graph\n");
    }
    fprintf(file, "timeInSeconds = %lf\n", out->timeInSeconds);
    fprintf(file, "numberNodes = %d\n", out->numberNodes);
    fprintf(file, "numberEdges = %d\n", out->edgesCount);

    if (writeAll)
    {
        for (int i = 0; i < (*out).numberNodes; i++)
        {
            if ((*out).hasNegativeCycle)
            {
                fprintf(file, "Predcessor of node %d is node %d\n", i,
                        (*out).predecessor[i]);
            }
            else
            {
                fprintf(file, "Distance from node %d to node = %d is %d\n",
                        (*out).startNode, i, (*out).dist[i]);
            }
        }
    }
    fclose(file);
}

int main(int argc, char **argv)
{
    BFOutput *result = bellmanFordCuda("../../data/graph_no_cycle_5.txt", 0);
    writeResult(result, "../../results/cuda/graph_no_cycle_5.txt", true);

    result = bellmanFordCuda("../../data/graph_cycle_5.txt", 0);
    writeResult(result, "../../results/cuda/graph_cycle_5.txt", true);
    freeBFOutput(result);
    return 0;
}