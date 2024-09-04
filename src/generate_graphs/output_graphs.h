#ifndef OUTPUT_GRAPHS_H
#define OUTPUT_GRAPHS_H

#include <stdio.h>
#include <stdlib.h>
#include "graph_structures.h" // Assuming this header defines SourceGraph and DestGraph

// Function declarations
void printSourceGraph(SourceGraph g);
void printDestGraph(DestGraph g);
void writeGraphToFile(DestGraph g, const char *filename);

#endif // OUTPUT_GRAPHS_H
