
/*
 * This is a CUDA version of bellman_ford algorithm
 * Compile: nvcc -arch=sm_52 -o cuda_bellman_ford cuda_bellman_ford.cu
 * Run: ./cuda_bellman_ford <input file> <number of blocks per grid> <number of threads per block>, you will find the
 * output file 'output.txt'
 * */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INF 1000000

typedef struct {
    bool hasNegativeCycle;
    int negativeCycleNode;
    double timeInSeconds;
    int numberNodes;
    int startNode;
    int *predecessor;
    int *dist;

} BFOutput;

typedef struct {
    int dest;
    int weight;
} SourceEdge;

typedef struct {
    int inNeighbours;  // count of edges
    int outNeighbours;
    SourceEdge *outEdges;
} SourceNode;

typedef struct {
    int numNodes;
    SourceNode *nodes;
} SourceGraph;

void readSourceGraphFromFileToDevice(const char *filename, int **neighbouringNodes, int **neighbouringNodesWeights,
                                     int *n, int *neighboursCount) {
    n = (int *)malloc(sizeof(int));
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }
    int numEdgesOut = 0;
    fscanf(file, "g %d\n", n);
    printf("Number of Nodes = %d\n", (*n));
    neighbouringNodes = (int **)malloc((*n) * sizeof(int *));
    neighbouringNodesWeights = (int **)malloc((*n) * sizeof(int *));
    neighboursCount = (int *)malloc((*n) * sizeof(int));

    for (int i = 0; i < (*n); i++) {
        int temp;
        fscanf(file, "n %d %d\n", &temp, &neighboursCount[i]);
        numEdgesOut = numEdgesOut + neighboursCount[i];
    }
    printf("Total number of edges = %d\n", numEdgesOut);

    for (int i = 0; i < (*n); i++) {
        int *tempNeighbours = (int *)malloc(neighboursCount[i] * sizeof(int));
        int *tempWeights = (int *)malloc(neighboursCount[i] * sizeof(int));

        for (int j = 0; j < neighboursCount[i]; j++) {
            int source;
            int dest;
            int weight;
            fscanf(file, "e %d %d %d\n", &source, &dest, &weight);
            tempNeighbours[j] = dest;
            tempWeights[j] = weight;
        }
        cudaMalloc(&neighbouringNodes[i], neighboursCount[i] * sizeof(int));
        cudaMemcpy(neighbouringNodes[i], tempNeighbours, neighboursCount[i] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&neighbouringNodesWeights[i], neighboursCount[i] * sizeof(int));
        cudaMemcpy(neighbouringNodesWeights[i], tempWeights, neighboursCount[i] * sizeof(int), cudaMemcpyHostToDevice);
    }

    fclose(file);
}

double gettime(void) {
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

__global__ void relax_initial(int *d_dist/*, int *d_predecessor, bool *d_wasUpdatedLastIter, bool *d_hasChanged, int n,
                              int startNode*/) {
    int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
    int i = bdim * bid + tid;
    int skip = bdim * gdim;
    printf("we are here");
    /*for (int k = i; k < n; k += skip) {
        d_predecessor[k] = -1;
        d_hasChanged[k] = false;
        if (k != startNode) {
            d_dist[k] = INF;
            d_wasUpdatedLastIter[k] = false;
        } else {
            d_dist[startNode] = 0;
            d_wasUpdatedLastIter[startNode] = true;
        }
    }
    __syncthreads();*/
}

__global__ void copyHasChanged(bool *wasUpdatedLastIter, bool *hasChanged, int n) {
    int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
    int i = bdim * bid + tid;
    int skip = bdim * gdim;

    for (int j = i; j < n; j += skip) {
        wasUpdatedLastIter[j] = hasChanged[j];
        wasUpdatedLastIter[j] = false;
    }
    __syncthreads();
}

__global__ void bellmanFordIteration(int* weights, int* neighbours, int neighboursCount, int n/*, int *predecessor, int *dist, bool *wasUpdatedLastIter,
                                     bool *hasChanged, int source */) {
    int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
    int i = bdim * bid + tid;
    int skip = bdim * gdim;
    printf("%d\n", weights[i]);
    /*for (int edgeIndex = i; edgeIndex < node.outNeighbours; edgeIndex += skip) {
        if (*wasUpdatedLastIter) {
            int destination = outEdges[edgeIndex].dest;
            int weight = outEdges[edgeIndex].weight;
            int new_dist = dist[source] + weight;
            if (new_dist < dist[destination]) {
                hasChanged[destination] = true;
                dist[destination] = new_dist;
                predecessor[destination] = source;
            }
        }
    }
    __syncthreads();*/
}

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param blockPerGrid number of blocks per grid
 * @param threadsPerBlock number of threads per block
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
 */
BFOutput *bellmanFordCuda(const char *filename, int startNode) {
    // Pointer to the graph on the device
    int **neighbouringNodes;
    int **neighbouringNodesWeights;
    int *n;
    int *neighboursCount;
    readSourceGraphFromFileToDevice(filename, neighbouringNodes, neighbouringNodesWeights, n, neighboursCount);
    int size = *n;
    printf("after read\n");

    int *h_dist = (int *)malloc(size * sizeof(int));
    int *d_dist;
    cudaMalloc(&d_dist, size * sizeof(int));

    int *h_predecessor = (int *)malloc(size * sizeof(int));
    int *d_predecessor;
    cudaMalloc(&d_predecessor, size * sizeof(int));

    bool *h_wasUpdatedLastIter = (bool *)malloc(size * sizeof(bool));
    bool *d_wasUpdatedLastIter;
    cudaMalloc(&d_wasUpdatedLastIter, size * sizeof(bool));

    bool *h_hasChanged = (bool *)malloc(size * sizeof(bool));
    bool *d_hasChanged;
    cudaMalloc(&d_hasChanged, size * sizeof(bool));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    // Call the function to copy the SourceGraph to the GPU
    printf("before dim3\n");

    dim3 gdim(blocksPerGrid);
    dim3 bdim(threadsPerBlock);
    double tstart, tend;
    tstart = gettime();
    printf("before relax_initial\n");
    relax_initial<<<gdim, bdim>>>(d_dist /*, d_predecessor, d_wasUpdatedLastIter, d_hasChanged, (*n), startNode*/);
    cudaDeviceSynchronize();  // wait for kernel to finish

    /*for (int iter = 0; iter < (*n); iter++) {
        for (int source = 0; source < (*n); ++source) {
            printf("before curnel\n");
            bellmanFordIteration<<<gdim, bdim>>>(neighbouringNodesWeights[iter], neighbouringNodes[iter],
    neighboursCount[iter], (*n),
                                                          (*d_graph).nodes[source].outNeighbours, d_predecessor, d_dist,
                                                          d_wasUpdatedLastIter, d_hasChanged, source);
            cudaDeviceSynchronize();
            /*if (iter == n - 1 && d_hasChanged) {
                tend = gettime();
                cudaMemcpy((*result).predecessor, d_predecessor, n * sizeof(int), cudaMemcpyDeviceToHost);
                (*result).negativeCycleNode = source;
                (*result).hasNegativeCycle = true;
                (*result).timeInSeconds = tend - tstart;
                return result;
            }
        }
        // copyHasChanged<<<gdim, bdim>>>(d_wasUpdatedLastIter, d_hasChanged, n);
        // cudaDeviceSynchronize();
    }

    /*tend = gettime();
    cudaMemcpy((*result).dist, d_dist, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((*result).predecessor, d_predecessor, n * sizeof(int), cudaMemcpyDeviceToHost);
    (*result).hasNegativeCycle = false;
    (*result).timeInSeconds = tend - tstart;*/
    return nullptr;
}

void writeResult(BFOutput *out, const char *filename, bool writeAll) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }
    if ((*out).hasNegativeCycle) {
        fprintf(file, "There is negative cycle in the graph\n");
        fprintf(file, "From this node one can bactrack to find the cycle = %d\n", (*out).negativeCycleNode);
    } else {
        fprintf(file, "There is NOT negative cycle in the graph\n");
    }
    fprintf(file, "timeInSeconds = %lf\n", (*out).timeInSeconds);
    fprintf(file, "numberNodes = %d\n", (*out).numberNodes);
    if (writeAll) {
        for (int i = 0; i < (*out).numberNodes; i++) {
            if ((*out).hasNegativeCycle) {
                fprintf(file, "Predcessor of node %d is node %d\n", i, (*out).predecessor[i]);
            } else {
                fprintf(file, "Distance from node %d to node = %d is %d\n", (*out).startNode, i, (*out).dist[i]);
            }
        }
    }
    fclose(file);
}

int main(int argc, char **argv) {
    BFOutput *result = bellmanFordCuda("../../data/no_cycle/graph_no_cycle_5.txt", 0);
    // printf("---------------- %d\n", (*result).hasNegativeCycle);
    /*writeResult(result, "../../results/omp_source/no_cycle/graph_no_cycle_5.txt", true);

    SourceGraph *readGraphNegativeCycle = readSourceGraphFromFile("../../data/cycle/graph_cycle_5.txt");
    BFOutput *resultCycle = bellmanFordCuda(blocksPerGrid, threadsPerBlock, readGraphNegativeCycle, 0);
    writeResult(resultCycle, "../../results/omp_source/cycle/graph_cycle_5.txt", true);*/
    return 0;
}