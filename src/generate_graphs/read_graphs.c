#include "read_graphs.h"

DestGraph readDestGraphFromFile(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    DestGraph g;
    fscanf(file, "g %d\n", &g.numNodes);
    g.nodes = (DestNode *)malloc(g.numNodes * sizeof(DestNode));
    for (int i = 0; i < g.numNodes; i++)
    {
        fscanf(file, "n %d %d\n", &g.nodes[i].inNeighbours, &g.nodes[i].outNeighbours);
        g.nodes[i].inEdges = (DestEdge *)malloc(g.nodes[i].inNeighbours * sizeof(DestEdge));
    }
    for (int i = 0; i < g.numNodes; i++)
    {
        for (int j = 0; j < g.nodes[i].inNeighbours; j++)
        {
            int x;
            fscanf(file, "e %d %d %d\n", &g.nodes[i].inEdges[j].source, &x, &g.nodes[i].inEdges[j].weight);
        }
    }

    fclose(file);
    return g;
}

SorceGraph readSorceGraphFromFile(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    SorceGraph g;
    fscanf(file, "g %d\n", &g.numNodes);
    g.nodes = (SorceNode *)malloc(g.numNodes * sizeof(SorceNode));
    int *indexeForNode = (int *)malloc(g.numNodes * sizeof(int));
    int numEdgesOut = 0, numEdgesIn = 0;
    for (int i = 0; i < g.numNodes; i++)
    {
        fscanf(file, "n %d %d\n", &g.nodes[i].inNeighbours, &g.nodes[i].outNeighbours);
        g.nodes[i].outEdges = (SorceEdge *)malloc(g.nodes[i].outNeighbours * sizeof(SorceEdge));
        indexeForNode[i] = 0;
        numEdgesOut = numEdgesOut + g.nodes[i].outNeighbours;
        numEdgesIn = numEdgesIn + g.nodes[i].inNeighbours;
    }
    printf("Are numEdgesOut == numEdgesIn: %d\n", numEdgesIn == numEdgesOut);
    printf("Total number of edges = %d\n", numEdgesIn);

    for (int i = 0; i < numEdgesIn; i++)
    {
        int sorce;
        int dest;
        int weight;
        fscanf(file, "e %d %d %d\n", &sorce, &dest, &weight);
        g.nodes[sorce].outEdges[indexeForNode[sorce]].weight = weight;
        g.nodes[sorce].outEdges[indexeForNode[sorce]].dest = dest;
        indexeForNode[sorce]++;
    }

    fclose(file);
    return g;
}
