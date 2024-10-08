#ifndef GRAPH_OPERATIONS_H
#define GRAPH_OPERATIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include "graph_structures.h" // Ensure these headers define DestGraph, SourceGraph, etc.
#include "output_graphs.h"
#include "compare_graphs.h"
#include "read_graphs.h"

#define MIN_NODES 1024    // 2^10
#define MAX_NODES 1048576 // 2^20

// Function to generate a random integer between min and max (inclusive)
int randInt(int min, int max);

// Function to create a graph without any negative cycle
DestGraph *createGraphNoNegativeCycle(int numNodes, int numNeighbours);

// Function to create a graph that contains a negative cycle
DestGraph *createGraphWithNegativeCycle(int numNodes, int numNeighbours);

#endif // GRAPH_OPERATIONS_H
