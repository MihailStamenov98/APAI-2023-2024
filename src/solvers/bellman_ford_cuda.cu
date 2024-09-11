
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
    int startNode;
    int *predecessor;
    int *dist;

} BFOutput;

void readSourceGraphFromFileToDevice(const char *filename,
                                     int **&d_neighbouringNodes,
                                     int **&d_neighbouringNodesWeights, int *&n,
                                     int *&neighboursCount)
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

    fscanf(file, "g %d\n", n);
    printf("n is %d\n", *n);
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
    printf("Total number of edges = %d\n", numEdgesOut);

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

__global__ void relax_initial(int *d_dist, int *d_predecessor, bool *d_hasChanged,
                              int n, int startNode, int maxVal)
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
__global__ void copyHasChanged(bool *hasChanged,
                               int n)
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
                                     int *dist,
                                     bool *hasChanged, int source)
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

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other
 * vertices.
 * @param blockPerGrid number of blocks per grid
 * @param threadsPerBlock number of threads per block
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative
 * cycles
 */
BFOutput *bellmanFordCuda(const char *filename, int startNode)
{
    // Pointer to the graph on the device
    int **d_neighbouringNodes;
    int **d_neighbouringNodesWeights;
    int *n;
    int *neighboursCount;
    readSourceGraphFromFileToDevice(filename, d_neighbouringNodes,
                                    d_neighbouringNodesWeights, n,
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

    BFOutput *result;
    result = (BFOutput *)malloc(sizeof(BFOutput));
    (*result).startNode = startNode;
    (*result).predecessor = (int *)malloc(size * sizeof(int));
    (*result).dist = (int *)malloc(size * sizeof(int));
    (*result).negativeCycleNode = -1;
    (*result).numberNodes = size;

    dim3 gdim(blocksPerGrid);
    dim3 bdim(threadsPerBlock);
    double tstart, tend;
    tstart = gettime();
    printf("before relax_initial\n");
    relax_initial<<<gdim, bdim>>>(d_dist, d_predecessor,
                                  d_hasChanged, size, startNode, INF);
    cudaMemcpy(wasUpdatedLastIter, d_hasChanged, size * sizeof(bool),
               cudaMemcpyDeviceToHost);
    wasUpdatedLastIter[startNode] = true;
    printf("\n");
    cudaDeviceSynchronize(); // wait for kernel to finish

    printf("%d\n", size);
    for (int iter = 0; iter < size; iter++)
    {
        printf("iteration is: %d\n", iter);

        for (int source = 0; source < (*n); ++source)
        {
            if (wasUpdatedLastIter[source])
            {
                printf("source is: %d\n", source);

                bellmanFordIteration<<<gdim, bdim>>>(d_neighbouringNodesWeights[source],
                                                     d_neighbouringNodes[source],
                                                     neighboursCount[source],
                                                     d_predecessor,
                                                     d_dist,
                                                     d_hasChanged,
                                                     source);
            }
            cudaDeviceSynchronize();
            if (iter == size - 1 && d_hasChanged)
            {
                printf("has neg cycle\n");
                tend = gettime();
                cudaMemcpy((*result).predecessor, d_predecessor, size * sizeof(int),
                           cudaMemcpyDeviceToHost);
                (*result).negativeCycleNode = source;
                (*result).hasNegativeCycle = true;
                (*result).timeInSeconds = tend - tstart;
                return result;
            }
        }
        cudaMemcpy(wasUpdatedLastIter, d_hasChanged, size * sizeof(bool),
                   cudaMemcpyDeviceToHost);
        copyHasChanged<<<gdim, bdim>>>(d_hasChanged, size);
        cudaDeviceSynchronize();
    }

    tend = gettime();
    cudaMemcpy((*result).dist, d_dist, size * sizeof(int), cudaMemcpyDeviceToHost);
    (*result).hasNegativeCycle = false;
    (*result).timeInSeconds = tend - tstart;

    /*cudaMemcpy(h_dist, d_dist, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_predecessor, d_predecessor, size * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wasUpdatedLastIter, d_wasUpdatedLastIter, size * sizeof(bool),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hasChanged, d_hasChanged, size * sizeof(bool),
               cudaMemcpyDeviceToHost);*/

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
    fprintf(file, "timeInSeconds = %lf\n", (*out).timeInSeconds);
    fprintf(file, "numberNodes = %d\n", (*out).numberNodes);
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
    BFOutput *result =
        bellmanFordCuda("../../data/no_cycle/graph_no_cycle_5.txt", 0);
    printf("bllman ford finished\n");
    printf("---------------- %d\n", (*result).hasNegativeCycle);
    //  writeResult(result, "../../results/omp_source/no_cycle/graph_no_cycle_5.txt",
    //  true);

    /*SourceGraph *readGraphNegativeCycle =
      readSourceGraphFromFile("../../data/cycle/graph_cycle_5.txt");
  BFOutput *resultCycle = bellmanFordCuda(blocksPerGrid, threadsPerBlock,
                                          readGraphNegativeCycle, 0);
  writeResult(resultCycle, "../../results/omp_source/cycle/graph_cycle_5.txt",
              true);
  */
    return 0;
}