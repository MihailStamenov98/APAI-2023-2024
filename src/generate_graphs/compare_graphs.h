#ifndef COMPARE_GRAPHS_H
#define COMPARE_GRAPHS_H

#include <stdbool.h>
#include <stdio.h>
#include "graph_structures.h" // Ensure this header defines DestGraph and SorceGraph

// Function to compare a DestGraph and a SorceGraph
// Returns true if the graphs are equivalent, false otherwise
bool compareGraphs(DestGraph g1, SorceGraph g2);
bool compareDestGraphs(DestGraph g1, DestGraph g2);

#endif // COMPARE_GRAPHS_H
