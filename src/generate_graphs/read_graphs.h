#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include <stdio.h>
#include <stdlib.h>
#include "graph_structures.h" // Assuming this is where the SourceGraph and DestGraph are defined

// Function declarations
DestGraph *readDestGraphFromFile(const char *filename);
SourceGraph *readSourceGraphFromFile(const char *filename);

#endif // GRAPH_IO_H
