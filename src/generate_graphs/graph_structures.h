// graph_structures.h

#ifndef GRAPH_STRUCTURES_H
#define GRAPH_STRUCTURES_H

typedef struct
{
    int source;
    int weight;
} DestEdge;

typedef struct
{
    int dest;
    int weight;
} SorceEdge;

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
    SorceEdge *outEdges;
} SorceNode;

typedef struct
{
    int numNodes;
    DestNode *nodes;
} DestGraph;

typedef struct
{
    int numNodes;
    SorceNode *nodes;
} SorceGraph;

#endif // GRAPH_STRUCTURES_H
