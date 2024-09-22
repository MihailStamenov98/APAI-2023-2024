/*
 * This is a openmp version of bellman_ford algorithm
 * Compile: g++ -fopenmp -o openmp_bellman_ford_parallel_basic openmp_bellman_ford.cpp
 * Run: ./openmp_bellman_ford <input file> <number of threads>, you will find the output file 'output.txt'
 * */
#include <stdbool.h>
#include "../generate_graphs/graph_generator.h"
#include "../generate_graphs/output_graphs.h"
#include "../generate_graphs/read_graphs.h"
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
BFOutput *bellmanFord(int p, DestGraph *g, int startNode)
{
    BFOutput *result = initBFOutput(startNode, g->numNodes, g->numEdges);

    double tstart, tstop;
    tstart = omp_get_wtime();

    bool *wasUpdatedLastIter = (bool *)malloc((*g).numNodes * sizeof(bool));
    bool *isUpdatedThisIter = (bool *)malloc((*g).numNodes * sizeof(bool));

    omp_set_num_threads(p);

    // initialize distances
#pragma omp parallel for
    for (int i = 0; i < g->numNodes; i++)
    {
        result->dist[i] = INF;
        result->predecessor[i] = -1;
        wasUpdatedLastIter[i] = false;
        isUpdatedThisIter[i] = false;
    }
    result->dist[startNode] = 0;
    wasUpdatedLastIter[startNode] = true;
    bool has_changed = false;
    for (int iter = 0; iter < (*g).numNodes; iter++)
    {
#pragma omp parallel for
        for (int i = 0; i < (*g).numNodes; i++)
        {
            isUpdatedThisIter[i] = false;
        }
#pragma omp parallel for reduction(| : has_changed) schedule(dynamic)
        for (int dest = 0; dest < (*g).numNodes; ++dest)
        { 
            // int thread = omp_get_thread_num();
            //  printf("Thread %d works on node %d\n", thread, dest);
            for (int j = 0; j < (*g).nodes[dest].inNeighbours; ++j)
            {
                int source = (*g).nodes[dest].inEdges[j].source;
                if (wasUpdatedLastIter[source])
                {

                    int weight = (*g).nodes[dest].inEdges[j].weight;
                    int new_dis = (*result).dist[source] + weight;
                    // printf("source = %d, dest = %d, weight= %d, dist_dest = %d, dist_source = %d, new_weight = %d\n", source, dest, weight, result.dist[dest], result.dist[source], new_dis);
                    if (new_dis < result->dist[dest])
                    {
                        isUpdatedThisIter[dest] = true;
                        has_changed = true;
                        (*result).dist[dest] = new_dis;
                        (*result).predecessor[dest] = source;
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
        }
        if (!has_changed)
        {
            (*result).hasNegativeCycle = false;
            tstop = omp_get_wtime();
            (*result).timeInSeconds = tstop - tstart;
            return result;
        }
        has_changed = false;

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
        DestGraph *readGraph = readDestGraphFromFile(filename);
        BFOutput *result = bellmanFord(power_of_two(i), readGraph, 0);
        snprintf(filename, sizeof(filename), "../../results/omp_dest/graph_no_cycle_%d.edg_%d.txt", numnodes, maxNumEdges);
        writeResult(result, filename, false);
        hasCicle[2 * i] = result->hasNegativeCycle;
        times[2 * i] = result->timeInSeconds;
        printf("First graph should not have cycle for no cycle: %d\n", result->hasNegativeCycle);
        printf("Time for no cycle: %f\n", result->timeInSeconds);
        freeBFOutput(result);
        freeDestGraph(readGraph);

        snprintf(filename, sizeof(filename), "../../data/graph_cycle_%d.edg_%d.txt", numnodes, maxNumEdges);
        readGraph = readDestGraphFromFile(filename);
        result = bellmanFord(power_of_two(i), readGraph, 0);
        snprintf(filename, sizeof(filename), "../../results/omp_dest/graph_cycle_%d.edg_%d.txt", numnodes, maxNumEdges);
        writeResult(result, filename, false);
        hasCicle[2 * i + 1] = result->hasNegativeCycle;
        times[2 * i + 1] = result->timeInSeconds;
        printf("Second graph should have cycle for no cycle: %d\n", result->hasNegativeCycle);
        printf("Time for cycle: %f\n", result->timeInSeconds);
        freeBFOutput(result);
        freeDestGraph(readGraph);
    }
    FILE *fileTimes = fopen("../../results/omp_dest/times.txt", "w");        // Open file in write mode
    FILE *fileHasCicle = fopen("../../results/omp_dest/has_cicle.txt", "w"); // Open file in write mode

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
    printf("OMP Dest ended\n");
    return 0;
}