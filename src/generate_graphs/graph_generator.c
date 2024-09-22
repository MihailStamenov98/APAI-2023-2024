#include "graph_generator.h"

int randInt(int min, int max)
{
    return min + rand() % (max - min + 1);
}

int getNumber(int i, int *lastNodeIndex, int *sourceForNodes)
{
    int source = i;
    int index;
    while (source == i)
    {
        index = randInt(0, *lastNodeIndex);
        source = sourceForNodes[index];
    }
    int temp = sourceForNodes[*lastNodeIndex];
    sourceForNodes[*lastNodeIndex] = sourceForNodes[index];
    sourceForNodes[index] = temp;
    (*lastNodeIndex)--;
    return source;
}

DestGraph *createGraphNoNegativeCycle(int numNodes, int numNeighbours)
{
    DestGraph *g;
    g = (DestGraph *)malloc(sizeof(DestGraph));
    g->numNodes = numNodes;
    g->numEdges = 0;
    printf("numNodes = %d\n", numNodes);
    g->nodes = (DestNode *)malloc(numNodes * sizeof(DestNode));
    int *sourceForNodes = (int *)malloc(numNodes * sizeof(int));

    for (int i = 0; i < (*g).numNodes; i++)
    {
        (*g).nodes[i].outNeighbours = 0;
        sourceForNodes[i] = i;
    }
    for (int i = 0; i < (*g).numNodes; i++)
    {
        (*g).nodes[i].inNeighbours = randInt(1, 6 * (numNeighbours / 10) + numNeighbours % 10);
        g->numEdges += (g->nodes[i].inNeighbours);
        (*g).nodes[i].inEdges = (DestEdge *)malloc((*g).nodes[i].inNeighbours * sizeof(DestEdge));
        int lastNodeIndex = numNodes - 1;
        for (int j = 0; j < (*g).nodes[i].inNeighbours; j++)
        {
            int source = getNumber(i, &lastNodeIndex, sourceForNodes);
            (*g).nodes[i].inEdges[j].source = source;
            (*g).nodes[source].outNeighbours++;
            (*g).nodes[i].inEdges[j].weight = randInt(1, 20);
        }
    }
    printf("Graph generated with number edges %d\n", g->numEdges);

    return g;
}

// Function to create a graph with a negative cycle
DestGraph *createGraphWithNegativeCycle(int numNodes, int numNeighbours)
{
    DestGraph *g = createGraphNoNegativeCycle(numNodes, numNeighbours);
    // Introduce a negative cycle
    int num_cycles = randInt(1, 100);
    for (int cycle = 0; cycle < num_cycles; cycle++)
    {
        int cycleLen = randInt(3, numNodes); // Create a small cycle of 3-5 nodes
        int cycleStart = randInt(0, numNodes - cycleLen);
        int *sourceForNodes = (int *)malloc(cycleLen * sizeof(int));
        int lastNodeIndex = cycleLen - 1;
        for (int i = 0; i < cycleLen; i++)
        {
            sourceForNodes[i] = cycleStart + i;
        }
        int currNodeInCycle = cycleStart;

        for (int i = 0; i < cycleLen; i++)
        {
            int inNeighboursCount = g->nodes[currNodeInCycle].inNeighbours;
            int source = getNumber(currNodeInCycle, &lastNodeIndex, sourceForNodes);

            int sourceIndex = randInt(0, inNeighboursCount - 1);
            int oldEdgeNodeID = (*g).nodes[currNodeInCycle].inEdges[sourceIndex].source;
            DestEdge newEdge = (DestEdge){source, -1};
            g->nodes[currNodeInCycle].inEdges[sourceIndex] = newEdge;
            g->nodes[oldEdgeNodeID].outNeighbours--;
            g->nodes[newEdge.source].outNeighbours++;
            currNodeInCycle = source;
        }
    }
    printf("Negative cycle added to graph");

    return g;
}
