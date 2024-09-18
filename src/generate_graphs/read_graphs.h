#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include <stdio.h>
#include <stdlib.h>
#include "graph_structures.h"

// Function declarations
DestGraph *readDestGraphFromFile(const char *filename);
SourceGraph *readSourceGraphFromFile(const char *filename);
int **readGraphToMatrix(const char *filename, int *numNodes);
void get_numbers(int index, int *num1, int *num2);
#endif // GRAPH_IO_H
