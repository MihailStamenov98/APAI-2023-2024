#include <cuda_runtime.h>
#include <stdio.h>
int getSmt(int x) { return x + 1; }
#define INF 1000000

// CUDA kernel to add 30 to each element
__global__ void addThirty(int *neighbouringNodes,
                          int *neighbouringNodesWeights,
                          int neighboursCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < neighboursCount)
    {
        printf("neighbouringNodes[%d] = %d\n", idx,
               neighbouringNodes[idx]);

        printf("neighbouringNodesWeights[%d] = %d\n", idx,
               neighbouringNodesWeights[idx]);
    }
}

__global__ void relax_initial(int *d_dist, int *d_predecessor,
                              bool *d_wasUpdatedLastIter, bool *d_hasChanged,
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
            d_wasUpdatedLastIter[k] = false;
        }
        else
        {
            d_dist[startNode] = 0;
            d_wasUpdatedLastIter[startNode] = true;
        }
    }
    __syncthreads();
}
void readSourceGraphFromFileToDevice(const char *filename,
                                     int **&neighbouringNodes,
                                     int **&neighbouringNodesWeights,
                                     int *&n,
                                     int *&neighboursCount)
{
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
    neighboursCount = (int *)malloc(*n * sizeof(int));
    int *indexeForNode = (int *)malloc(*n * sizeof(int));
    int numEdgesOut = 0;
    for (int i = 0; i < *n; i++)
    {
        int temp;
        fscanf(file, "n %d %d\n", &temp, &neighboursCount[i]);
        neighbouringNodes[i] = (int *)malloc(neighboursCount[i] * sizeof(int));
        neighbouringNodesWeights[i] = (int *)malloc(neighboursCount[i] * sizeof(int));
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
    fclose(file);
}

int main()
{

    // Pointer to the graph on the device
    int **neighbouringNodes;
    int **neighbouringNodesWeights;
    int *n;
    int *neighboursCount;
    readSourceGraphFromFileToDevice("data/no_cycle/graph_no_cycle_5.txt",
                                    neighbouringNodes,
                                    neighbouringNodesWeights,
                                    n,
                                    neighboursCount);
    /*printf("here %d\n", (n));
    printf("here %d\n", (*n));

    for (int i = 0; i < (*n); ++i)
    {
        printf("source %d", i);
        for (int j = 0; j < neighboursCount[i]; j++)
        {
            printf("dest %d\n", neighbouringNodes[i][j]);
            printf("weight %d\n", neighbouringNodesWeights[i][j]);
        }
    }*/
    int size = *n;
    int *d_neighbouringNodes = (int *)malloc(size * sizeof(int));
    int *d_neighbouringNodesWeights = (int *)malloc(size * sizeof(int));
    cudaMalloc(&d_neighbouringNodes, neighboursCount[0] * sizeof(int));
    cudaMalloc(&d_neighbouringNodesWeights, neighboursCount[0] * sizeof(int));
    cudaMemcpy(d_neighbouringNodes, neighbouringNodes[0], neighboursCount[0] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbouringNodesWeights, neighbouringNodesWeights[0], neighboursCount[0] * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (5 + threadsPerBlock - 1) / threadsPerBlock;
    printf("%d\n", neighboursCount[0]);

    addThirty<<<blocksPerGrid, threadsPerBlock>>>(d_neighbouringNodes, d_neighbouringNodesWeights, neighboursCount[0]);
    cudaDeviceSynchronize(); // wait for kernel to finish*/

    return 0;
}
