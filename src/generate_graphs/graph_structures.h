// graph_structures.h

#ifndef GRAPH_STRUCTURES_H
#define GRAPH_STRUCTURES_H
#include <stdlib.h>

typedef struct
{
    int source;
    int weight;
} DestEdge;

typedef struct
{
    int dest;
    int weight;
} SourceEdge;

typedef struct
{
    int inNeighbours; // count of edges
    int outNeighbours;
    DestEdge *inEdges;
} DestNode;

typedef struct
{
    int inNeighbours; // count of edges
    int outNeighbours;
    SourceEdge *outEdges;
} SourceNode;

typedef struct
{
    int numNodes;
    int numEdges;
    DestNode *nodes;
} DestGraph;

typedef struct
{
    int numNodes;
    int numEdges;
    SourceNode *nodes;
} SourceGraph;
void freeDestGraph(DestGraph *graph);
void freeSourceGraph(SourceGraph *graph);
#endif // GRAPH_STRUCTURES_H
