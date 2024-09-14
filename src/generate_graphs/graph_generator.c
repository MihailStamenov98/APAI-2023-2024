#include "graph_generator.h"

int randInt(int min, int max)
{
    return min + rand() % (max - min + 1);
}

DestGraph *createGraphNoNegativeCycle(int numNodes, int numNeighbours)
{
    DestGraph *g;
    g = (DestGraph *)malloc(sizeof(DestGraph));
    g->numNodes = numNodes;
    g->numEdges = 0;
    printf("numNodes = %d\n", numNodes);
    (*g).nodes = (DestNode *)malloc(numNodes * sizeof(DestNode));
    int *sourceForNodes = (int *)malloc(numNodes * sizeof(int));

    for (int i = 0; i < (*g).numNodes; i++)
    {
        (*g).nodes[i].outNeighbours = 0;
        sourceForNodes[i] = i;
    }
    for (int i = 0; i < (*g).numNodes; i++)
    {
        (*g).nodes[i].inNeighbours = randInt(1, numNeighbours);
        g->numEdges += (g->nodes[i].inNeighbours);
        (*g).nodes[i].inEdges = (DestEdge *)malloc((*g).nodes[i].inNeighbours * sizeof(DestEdge));
        int lastNodeIndex = numNodes - 1;
        for (int j = 0; j < (*g).nodes[i].inNeighbours; j++)
        {
            int source = i;
            int index;
            while (source == i)
            {
                index = randInt(0, lastNodeIndex);
                source = sourceForNodes[index];
            }
            int temp = sourceForNodes[lastNodeIndex];
            sourceForNodes[lastNodeIndex] = sourceForNodes[index];
            sourceForNodes[index] = temp;
            lastNodeIndex--;
            (*g).nodes[i].inEdges[j].source = source;
            (*g).nodes[source].outNeighbours++;
            (*g).nodes[i].inEdges[j].weight = randInt(0, 20);
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
    int cycleLen = randInt(3, numNodes); // Create a small cycle of 3-5 nodes
    int cycleStart = randInt(0, numNodes - cycleLen);
    for (int i = 0; i < cycleLen; i++)
    {
        int currNodeInNeighbours = (*g).nodes[cycleStart + i].inNeighbours;
        int sourceIndex = randInt(0, currNodeInNeighbours - 1);
        int oldEdgeNodeID = (*g).nodes[cycleStart + i].inEdges[sourceIndex].source;
        DestEdge newEdge = (DestEdge){cycleStart + (i + 1) % cycleLen, -1};
        g->nodes[cycleStart + i].inEdges[sourceIndex] = newEdge;
        g->nodes[oldEdgeNodeID].outNeighbours--;
        g->nodes[newEdge.source].outNeighbours++;
    }

    return g;
}
