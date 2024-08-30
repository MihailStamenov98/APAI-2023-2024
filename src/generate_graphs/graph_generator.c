#include "graph_generator.h"

int randInt(int min, int max)
{
    return min + rand() % (max - min + 1);
}

DestGraph createGraphNoNegativeCycle(int numNodes)
{
    DestGraph g;
    g.numNodes = numNodes;
    printf("numNodes = %d\n", numNodes);
    g.nodes = (DestNode *)malloc(numNodes * sizeof(DestNode));
    for (int i = 0; i < g.numNodes; i++)
    {
        g.nodes[i].outNeighbours = 0;
    }
    for (int i = 0; i < g.numNodes; i++)
    {
        g.nodes[i].inNeighbours = randInt(1, 3);
        g.nodes[i].inEdges = (DestEdge *)malloc(g.nodes[i].inNeighbours * sizeof(DestEdge));
        for (int j = 0; j < g.nodes[i].inNeighbours; j++)
        {
            g.nodes[i].inEdges[j].source = randInt(0, numNodes - 1);
            g.nodes[g.nodes[i].inEdges[j].source].outNeighbours++;
            g.nodes[i].inEdges[j].weight = randInt(-10, 20);
        }
    }
    printf("Graph generated\n");

    return g;
}

// Function to create a graph with a negative cycle
DestGraph createGraphWithNegativeCycle(int numNodes)
{
    DestGraph g = createGraphNoNegativeCycle(numNodes);
    // Introduce a negative cycle
    int cycleLen = randInt(3, 200); // Create a small cycle of 3-5 nodes
    int cycleStart = randInt(0, numNodes - cycleLen);
    for (int i = 0; i < cycleLen; i++)
    {
        int currNodeInNeighbours = g.nodes[cycleStart + i].inNeighbours;
        int sourceIndex = randInt(0, currNodeInNeighbours - 1);
        int oldEdgeNodeID = g.nodes[cycleStart + i].inEdges[sourceIndex].source;
        DestEdge newEdge = (DestEdge){cycleStart + (i + 1) % cycleLen, -1};
        g.nodes[cycleStart + i].inEdges[sourceIndex] = newEdge;
        g.nodes[oldEdgeNodeID].outNeighbours--;
        g.nodes[newEdge.source].outNeighbours++;
    }

    return g;
}

void freeDestGraph(DestGraph g)
{
    for (int i = 0; i < g.numNodes; ++i)
    {
        free(g.nodes[i].inEdges);
    }

    free(g.nodes);
}
void freeSorceGraph(SorceGraph g)
{
    for (int i = 0; i < g.numNodes; ++i)
    {
        free(g.nodes[i].outEdges);
    }

    free(g.nodes);
}

int main()
{
    int size = 10;
    DestGraph destGNoCycle = createGraphNoNegativeCycle(size);
    char filenameNoCycle[50];
    snprintf(filenameNoCycle, sizeof(filenameNoCycle), "graph_no_cycle_%d.txt", size);
    writeGraphToFile(destGNoCycle, filenameNoCycle);

    DestGraph readDestG = readDestGraphFromFile(filenameNoCycle);
    printf("%d\n", compareDestGraphs(destGNoCycle, readDestG));

    SorceGraph readSorceG = readSorceGraphFromFile(filenameNoCycle);
    // printSorceGraph(readSorceG);
    printf("%d\n", compareGraphs(destGNoCycle, readSorceG));

    freeDestGraph(readDestG);
    freeDestGraph(destGNoCycle);
    freeSorceGraph(readSorceG);
    return 0;
}