/*
 * This is a openmp version of bellman_ford algorithm
 * Compile: g++ -fopenmp -o openmp_bellman_ford_parallel_basic openmp_bellman_ford.cpp
 * Run: ./openmp_bellman_ford <input file> <number of threads>, you will find the output file 'output.txt'
 * */
#include <stdbool.h>
#include "../generate_graphs/graph_generator.h"
#include "../generate_graphs/output_graphs.h"

#define INF 1000000

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param p number of processes
 * @param g input destination graph
 * @param *dist distance array
 * @param *hasNegativeCycle a bool variable to recode if there are negative cycles
 * @param *negativeCycleNode a bool variable to recode the node to strat the search from
 */
void bellmanFord(int p, DestGraph g, int *dist, bool *hasNegativeCycle, int *negativeCycleNode)
{
    int *predecessor = (int *)malloc(g.numNodes * sizeof(int));

    // initialize distances
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < g.numNodes; i++)
    {
        dist[i] = INF;
        predecessor[i] = -1;
    }
    dist[0] = 0;
    bool has_changed = false;
    for (int iter = 0; iter < g.numNodes; iter++)
    {
#pragma omp parallel for reduction(| : has_changed)
        for (int dest = 0; dest < g.numNodes; ++dest)
        {
            for (int j = 0; j < g.nodes[dest].inNeighbours; ++j)
            {
                int source = g.nodes[dest].inEdges[j].source;
                int weight = g.nodes[dest].inEdges[j].weight;
                int new_dis = dist[source] + weight;
                if (new_dis < dist[dest])
                {
                    has_changed = true;
                    dist[dest] = new_dis;
                    predecessor[dest] = source;
                    if (iter == g.numNodes - 1)
                    {
#pragma omp critical
                        {
                            *negativeCycleNode = source;
                            *hasNegativeCycle = true;
                        }
                    }
                }
            }
        }
        if (!has_changed)
        {
            *hasNegativeCycle = false;
            return;
        }
    }

    // step 4: free memory (if any)
}

int main(int argc, char **argv)
{
    /*DestGraph readGraphNegativeCycle = readDestGraphFromFile("negative_cycle.txt");
    int *dist = (int *)malloc(readGraphNegativeCycle.numNodes * sizeof(int));
    int *node = (int *)malloc(sizeof(int));
    *node = -1;
    bool *hasNegativeCycle = (bool *)malloc(sizeof(bool));
    // printDestGraph(readGraphNegativeCycle);
    bellmanFord(1, readGraphNegativeCycle, dist, hasNegativeCycle, node);
    printf("Is there negative cycle: %d\n the node to start from is : %d\n", *hasNegativeCycle, *node);*/
    DestGraph readGraph = readDestGraphFromFile("graph_no_cycle_5.txt");
    int *distNoCycle = (int *)malloc(readGraph.numNodes * sizeof(int));
    int *nodeNoCycle = (int *)malloc(sizeof(int));
    *nodeNoCycle = -1;
    bool *hasNegativeCycleNoCycle = (bool *)malloc(sizeof(bool));
    // printDestGraph(readGraphNegativeCycle);
    bellmanFord(2, readGraph, distNoCycle, hasNegativeCycleNoCycle, nodeNoCycle);
    printf("Is there negative cycle: %d\n the node to start from is : %d\n", *hasNegativeCycleNoCycle, *nodeNoCycle);
    for (int i = 0; i < readGraph.numNodes; ++i)
    {
        printf("Node %d has dist = %d\n", i, distNoCycle[i]);
    }

    DestGraph readGraphNegativeCycle = readDestGraphFromFile("graph_cycle_5.txt");
    int *distCycle = (int *)malloc(readGraphNegativeCycle.numNodes * sizeof(int));
    int *nodeCycle = (int *)malloc(sizeof(int));
    *nodeCycle = -1;
    bool *hasNegativeCycle = (bool *)malloc(sizeof(bool));
    // printDestGraph(readGraphNegativeCycle);
    bellmanFord(2, readGraphNegativeCycle, distCycle, hasNegativeCycle, nodeCycle);
    printf("Is there negative cycle: %d\n the node to start from is : %d\n", *hasNegativeCycle, *nodeCycle);
    return 0;
}