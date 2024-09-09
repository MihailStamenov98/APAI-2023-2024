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
BFOutput* bellmanFordSource(int p, SourceGraph* g, int startNode)
{
    BFOutput* result;
    result = (BFOutput *)malloc(sizeof(BFOutput));

    double tstart, tstop;
    tstart = omp_get_wtime();

    (*result).startNode = startNode;
    (*result).predecessor = (int *)malloc((*g).numNodes * sizeof(int));
    (*result).dist = (int *)malloc((*g).numNodes * sizeof(int));
    (*result).negativeCycleNode = -1;
    (*result).numberNodes = (*g).numNodes;
    (*result).hasNegativeCycle = false;
    omp_set_num_threads(p);
    bool *wasUpdatedLastIter = (bool *)malloc((*g).numNodes * sizeof(bool));
    bool *isUpdatedThisIter = (bool *)malloc((*g).numNodes * sizeof(bool));
    // initialize distances
#pragma omp parallel for
    for (int i = 0; i < (*g).numNodes; i++)
    {
        (*result).dist[i] = INF;
        (*result).predecessor[i] = -1;
        wasUpdatedLastIter[i] = false;
        isUpdatedThisIter[i] = false;
    }
    (*result).dist[startNode] = 0;
    wasUpdatedLastIter[startNode] = true;

    for (int iter = 0; iter < (*g).numNodes; iter++)
    {
#pragma omp parallel for
        for (int source = 0; source < (*g).numNodes; ++source)
        {
            isUpdatedThisIter[source] = false;
        }
        for (int source = 0; source < (*g).numNodes; ++source)
        {
#pragma omp parallel for
            for (int edgeIndex = 0; edgeIndex < (*g).nodes[source].outNeighbours; ++edgeIndex)
            {
                if (wasUpdatedLastIter[source])
                {
                    int destination = (*g).nodes[source].outEdges[edgeIndex].dest;
                    int weight = (*g).nodes[source].outEdges[edgeIndex].weight;
                    int new_dist = (*result).dist[source] + weight;
                    // printf("source = %d, dest = %d, weight= %d, dist_dest = %d, dist_source = %d, new_weight = %d\n", source, dest, weight, result.dist[dest], result.dist[source], new_dis);
                    if (new_dist < (*result).dist[destination])
                    {
                        isUpdatedThisIter[destination] = true;
                        (*result).dist[destination] = new_dist;
                        (*result).predecessor[destination] = source;
                        if (iter == (*g).numNodes - 1)
                        {

#pragma omp critical
                            {
                                (*result).negativeCycleNode = source;
                                (*result).hasNegativeCycle = true;
                            }
                        }
                    }
                }
            }
            if ((*result).hasNegativeCycle)
            {
                return result;
            }
        }
        bool isThereChangeInIteration = false;
#pragma omp parallel for reduction(| : isThereChangeInIteration)
        for (int source = 0; source < (*g).numNodes; ++source)
        {
            wasUpdatedLastIter[source] = isUpdatedThisIter[source];
            isThereChangeInIteration = isThereChangeInIteration | isUpdatedThisIter[source];
        }

        if (!isThereChangeInIteration)
        {
            (*result).hasNegativeCycle = false;
            tstop = omp_get_wtime();
            (*result).timeInSeconds = tstop - tstart;
            return result;
        }

#pragma omp parallel for
        for (int i = 0; i < (*g).numNodes; i++)
        {
            wasUpdatedLastIter[i] = isUpdatedThisIter[i];
        }
    }

    tstop = omp_get_wtime();
    (*result).timeInSeconds = tstop - tstart;
    return result;
}

int main()
{
    SourceGraph* readGraph = readSourceGraphFromFile("../../data/no_cycle/graph_no_cycle_5.txt");
    BFOutput* result = bellmanFordSource(2, readGraph, 0);
    printf("---------------- %d\n", (*result).hasNegativeCycle);
    writeResult(result, "../../results/omp_source/no_cycle/graph_no_cycle_5.txt", true);

    SourceGraph* readGraphNegativeCycle = readSourceGraphFromFile("../../data/cycle/graph_cycle_5.txt");
    BFOutput* resultCycle = bellmanFordSource(2, readGraphNegativeCycle, 0);
    writeResult(resultCycle, "../../results/omp_source/cycle/graph_cycle_5.txt", true);
    return 0;
}