#ifndef OUTPUT_STRUCTURES_H
#define OUTPUT_STRUCTURES_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct
{
    bool hasNegativeCycle;
    int negativeCycleNode;
    double timeInSeconds;
    int numberNodes;
    int startNode;
    int *predecessor;
    int *dist;

} BFOutput;

void writeResult(BFOutput* out, const char *filename, bool writeAll);
void printResult(BFOutput* out, bool writeAll);

#endif
