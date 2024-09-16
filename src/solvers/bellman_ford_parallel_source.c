/*
 * This is a openmp version of bellman_ford algorithm
 * Compile: g++ -fopenmp -o openmp_bellman_ford_parallel_basic
 * openmp_bellman_ford.cpp Run: ./openmp_bellman_ford <input file> <number of
 * threads>, you will find the output file 'output.txt'
 * */
#include "../generate_graphs/graph_generator.h"
#include "../generate_graphs/output_graphs.h"
#include "output_structure.h"
#include <omp.h>
#include <stdbool.h>

#define INF 1000000

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other
 * vertices.
 * @param p number of processes
 * @param g input destination graph
 * @param *dist distance array
 * @param *hasNegativeCycle a bool variable to recode if there are negative
 * cycles
 * @param *negativeCycleNode a bool variable to recode the node to strat the
 * search from
 */
BFOutput *bellmanFordSource(int p, SourceGraph *g, int startNode)
{
    BFOutput *result = initBFOutput(startNode, g->numNodes, g->numEdges);

    double tstart, tstop;
    tstart = omp_get_wtime();

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
    bool isThereChangeInIteration;

    for (int iter = 0; iter < (*g).numNodes; iter++)
    {
        isThereChangeInIteration = false;
        for (int source = 0; source < (*g).numNodes; ++source)
        {
            if (wasUpdatedLastIter[source])
            {
#pragma omp parallel for
                for (int edgeIndex = 0; edgeIndex < (*g).nodes[source].outNeighbours;
                     ++edgeIndex)
                {

                    int destination = (*g).nodes[source].outEdges[edgeIndex].dest;
                    int weight = (*g).nodes[source].outEdges[edgeIndex].weight;
                    int new_dist = result->dist[source] + weight;
                    if (new_dist < result->dist[destination])
                    {
                        isUpdatedThisIter[destination] = true;
                        result->dist[destination] = new_dist;
                        result->predecessor[destination] = source;
                    }
                }
            }
#pragma omp parallel for reduction(| : isThereChangeInIteration)
            for (int i = 0; i < g->numNodes; ++i)
            {
                isThereChangeInIteration = isThereChangeInIteration | isUpdatedThisIter[i];
            }
            if (iter == g->numNodes - 1 && isThereChangeInIteration)
            {
                result->negativeCycleNode = source;
                result->hasNegativeCycle = true;
            }
        }

#pragma omp parallel for reduction(| : isThereChangeInIteration)
        for (int i = 0; i < g->numNodes; ++i)
        {
            wasUpdatedLastIter[i] = isUpdatedThisIter[i];
            isUpdatedThisIter[i] = false;
        }

        if (!isThereChangeInIteration)
        {
            result->hasNegativeCycle = false;
            tstop = omp_get_wtime();
            result->timeInSeconds = tstop - tstart;
            return result;
        }
    }

    tstop = omp_get_wtime();
    (*result).timeInSeconds = tstop - tstart;
    return result;
}

int main()
{
    SourceGraph *readGraph =
        readSourceGraphFromFile("../../data/graph_no_cycle_5.txt");
    BFOutput *result = bellmanFordSource(2, readGraph, 0);
    printf("---------------- %d\n", (*result).hasNegativeCycle);
    writeResult(result, "../../results/omp_source/graph_no_cycle_5.txt", true);

    SourceGraph *readGraphNegativeCycle =
        readSourceGraphFromFile("../../data/graph_cycle_5.txt");
    BFOutput *resultCycle = bellmanFordSource(2, readGraphNegativeCycle, 0);
    writeResult(resultCycle, "../../results/omp_source/graph_cycle_5.txt", true);
    freeBFOutput(result);
    freeBFOutput(resultCycle);
    freeSourceGraph(readGraph);
    freeSourceGraph(readGraphNegativeCycle);
    return 0;
}