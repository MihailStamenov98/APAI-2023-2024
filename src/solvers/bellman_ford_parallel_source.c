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
    int numnodes, maxNumEdges;
    bool hasCicle[18];
    double times[18];

    for (int i = 0; i < 9; i++)
    {
        get_numbers(i, &numnodes, &maxNumEdges);
        if (maxNumEdges == numnodes)
        {
            maxNumEdges = maxNumEdges - 1;
        }
        printf("For index = %d, numbers are %d, %d\n", i, numnodes, maxNumEdges);

        char filename[70];
        snprintf(filename, sizeof(filename), "../../data/graph_no_cycle_%d.edg_%d.txt", numnodes, maxNumEdges);
        SourceGraph *readGraph = readSourceGraphFromFile(filename);
        BFOutput *result = bellmanFordSource(power_of_two(i), readGraph, 0);
        snprintf(filename, sizeof(filename), "../../results/omp_source/graph_no_cycle_%d.edg_%d.txt", numnodes, maxNumEdges);
        writeResult(result, filename, true);
        hasCicle[2 * i] = result->hasNegativeCycle;
        times[2 * i] = result->timeInSeconds;
        printf("First graph should not have cycle for no cycle: %d\n", result->hasNegativeCycle);
        printf("Time for no cycle: %f\n", result->timeInSeconds);
        freeBFOutput(result);
        freeSourceGraph(readGraph);

        snprintf(filename, sizeof(filename), "../../data/graph_cycle_%d.edg_%d.txt", numnodes, maxNumEdges);
        readGraph = readSourceGraphFromFile(filename);
        result = bellmanFordSource(power_of_two(i), readGraph, 0);
        snprintf(filename, sizeof(filename), "../../results/omp_source/graph_cycle_%d.edg_%d.txt", numnodes, maxNumEdges);
        writeResult(result, filename, true);
        hasCicle[2 * i + 1] = result->hasNegativeCycle;
        times[2 * i + 1] = result->timeInSeconds;
        printf("Second graph should have cycle for no cycle: %d\n", result->hasNegativeCycle);
        printf("Time for cycle: %f\n", result->timeInSeconds);
        freeBFOutput(result);
        freeSourceGraph(readGraph);
    }
    FILE *fileTimes = fopen("../../results/omp_source/times.txt", "w");        // Open file in write mode
    FILE *fileHasCicle = fopen("../../results/omp_source/has_cicle.txt", "w"); // Open file in write mode

    if (fileTimes == NULL)
    {
        printf("Error opening file times.txt!\n");
        return 0;
    }
    if (fileHasCicle == NULL)
    {
        printf("Error opening file has_cicle.txt!\n");
        return 0;
    }

    for (int i = 0; i < 18; i++)
    {
        fprintf(fileTimes, "%f\n", times[i]);       // Write each integer to a new line
        fprintf(fileHasCicle, "%d\n", hasCicle[i]); // Write each integer to a new line
    }

    fclose(fileTimes);
    fclose(fileHasCicle);
    printf("OMP Source finished\n");
    return 0;
}