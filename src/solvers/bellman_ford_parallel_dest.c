/*
 * This is a openmp version of bellman_ford algorithm
 * Compile: g++ -fopenmp -o openmp_bellman_ford_parallel_basic openmp_bellman_ford.cpp
 * Run: ./openmp_bellman_ford <input file> <number of threads>, you will find the output file 'output.txt'
 * */
#include <stdbool.h>
#include "../generate_graphs/graph_generator.h"
#include "../generate_graphs/output_graphs.h"
#include <omp.h>
#include "output_structure.h"
#define INF 1000000

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param p number of processes
 * @param g input destination graph
 * @param *dist distance array
 * @param *hasNegativeCycle a bool variable to recode if there are negative cycles
 * @param *negativeCycleNode a bool variable to recode the node to strat the search from
 */
BFOutput bellmanFord(int p, DestGraph g, int startNode)
{
    BFOutput result;
    double tstart, tstop;
    tstart = omp_get_wtime();

    result.startNode = startNode;
    result.predecessor = (int *)malloc(g.numNodes * sizeof(int));
    result.dist = (int *)malloc(g.numNodes * sizeof(int));
    result.negativeCycleNode = -1;
    result.numberNodes = g.numNodes;

    omp_set_num_threads(p);

    // initialize distances
#pragma omp parallel for
    for (int i = 0; i < g.numNodes; i++)
    {
        result.dist[i] = INF;
        result.predecessor[i] = -1;
    }
    result.dist[startNode] = 0;
    bool has_changed = false;
    for (int iter = 0; iter < g.numNodes; iter++)
    {
#pragma omp parallel for reduction(| : has_changed) schedule(dynamic)
        for (int dest = 0; dest < g.numNodes; ++dest)
        {
            // int thread = omp_get_thread_num();
            //  printf("Thread %d works on node %d\n", thread, dest);
            for (int j = 0; j < g.nodes[dest].inNeighbours; ++j)
            {
                int source = g.nodes[dest].inEdges[j].source;
                if (result.dist[source] != INF)
                {

                    int weight = g.nodes[dest].inEdges[j].weight;
                    int new_dis = result.dist[source] + weight;
                    // printf("sorce = %d, dest = %d, weight= %d, dist_dest = %d, dist_source = %d, new_weight = %d\n", source, dest, weight, result.dist[dest], result.dist[source], new_dis);
                    if (new_dis < result.dist[dest])
                    {
                        has_changed = true;
                        result.dist[dest] = new_dis;
                        result.predecessor[dest] = source;
                        if (iter == g.numNodes - 1)
                        {

#pragma omp critical
                            {
                                result.negativeCycleNode = source;
                                result.hasNegativeCycle = true;
                            }
                        }
                    }
                }
            }
        }
        if (!has_changed)
        {
            result.hasNegativeCycle = false;
            tstop = omp_get_wtime();
            result.timeInSeconds = tstop - tstart;
            return result;
        }
        has_changed = false;
    }
    tstop = omp_get_wtime();
    result.timeInSeconds = tstop - tstart;
    return result;
}

int main()
{
    DestGraph readGraph = readDestGraphFromFile("../../data/no_cycle/graph_no_cycle_5.txt");
    BFOutput result = bellmanFord(2, readGraph, 0);
    printf("---------------- %d\n", result.hasNegativeCycle);
    writeResult(result, "../../results/omp_dest/no_cycle/graph_no_cycle_5.txt", true);
    DestGraph readGraphNegativeCycle = readDestGraphFromFile("../../data/cycle/graph_cycle_5.txt");
    BFOutput resultCycle = bellmanFord(2, readGraphNegativeCycle, 0);
    writeResult(resultCycle, "../../results/omp_dest/cycle/graph_cycle_5.txt", true);
    return 0;
}