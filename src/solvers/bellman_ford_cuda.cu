
/*
 * This is a CUDA version of bellman_ford algorithm
 * Compile: nvcc -arch=sm_52 -o cuda_bellman_ford cuda_bellman_ford.cu
 * Run: ./cuda_bellman_ford <input file> <number of blocks per grid> <number of threads per block>, you will find the
 * output file 'output.txt'
 * */
#include <stdbool.h>

#include "../generate_graphs/graph_generator.h"
#include "../generate_graphs/graph_structures.h"
#include "../generate_graphs/output_graphs.h"
#include "cuda_utils.h"
#include "output_structure.h"

#define INF 1000000

__global__ void relax_initial(int *d_dist, int *d_predecessor, bool *d_wasUpdatedLastIter, bool *d_hasChanged, int n) {
    int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
    int i = bdim * bid + tid;
    int skip = bdim * gdim;
    for (int k = i; k < n; k += skip) {
        d_dist[k] = INF;
        d_predecessor[k] = -1;
        d_wasUpdatedLastIter[k] = false;
        d_hasChanged[k] = false;
    }
    __syncthreads();
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

__global__ void bellmanFordIteration(SourceEdge *outEdges, int outNeighbours, int *predecessor, int *dist,
                                     bool *wasUpdatedLastIter, bool *hasChanged, int source) {
    int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
    int i = bdim * bid + tid;
    int skip = bdim * gdim;

    for (int edgeIndex = i; edgeIndex < outNeighbours; edgeIndex += skip) {
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
    __syncthreads();
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
BFOutput bellmanFordCuda(int blocksPerGrid, int threadsPerBlock, SourceGraph *g, int startNode) {
    // Pointer to the graph on the device
    SourceGraph *d_graph;
    int *d_dist;
    int *d_predecessor;
    bool *d_wasUpdatedLastIter, *d_hasChanged;
    int n = (*g).numNodes;
    cudaMalloc(&d_dist, n * sizeof(int));
    cudaMalloc(&d_predecessor, n * sizeof(int));
    cudaMalloc(&d_wasUpdatedLastIter, n * sizeof(bool));
    cudaMalloc(&d_hasChanged, n * sizeof(bool));
    // Call the function to copy the SourceGraph to the GPU
    copySourceGraphToDevice(g, &d_graph);

    dim3 gdim(blocksPerGrid);
    dim3 bdim(threadsPerBlock);
    relax_initial<<<gdim, bdim>>>(d_dist, d_predecessor, d_wasUpdatedLastIter, d_hasChanged, d_graph.numNodes);
    d_dist[startNode] = 0;
    d_wasUpdatedLastIter[startNode] = true;

    for (int iter = 0; iter < g.numNodes; iter++) {
        for (int source = 0; source < g.numNodes; ++source) {
            bellmanFordIteration<<<gdim, bdim>>>(d_graph[source].outEdges, d_graph[source].outNeighbours, d_predecessor,
                                                 d_dist, d_wasUpdatedLastIter, *d_hasChanged, source);
            if (iter == g.numNodes - 1 && d_hasChanged) {
            }
        }
        copyHasChanged<<<gdim, bdim>>>(d_wasUpdatedLastIter, d_hasChanged, n);
    }
    return result
}

int main(int argc, char **argv) {
    SourceGraph readGraph = readSourceGraphFromFile("../../data/no_cycle/graph_no_cycle_5.txt");
    BFOutput result = bellmanFordCuda(2, readGraph, 0);
    printf("---------------- %d\n", result.hasNegativeCycle);
    writeResult(result, "../../results/omp_source/no_cycle/graph_no_cycle_5.txt", true);

    SourceGraph readGraphNegativeCycle = readSourceGraphFromFile("../../data/cycle/graph_cycle_5.txt");
    BFOutput resultCycle = bellmanFordCuda(2, readGraphNegativeCycle, 0);
    writeResult(resultCycle, "../../results/omp_source/cycle/graph_cycle_5.txt", true);
    return 0;
}