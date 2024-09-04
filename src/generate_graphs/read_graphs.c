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

SourceGraph readSourceGraphFromFile(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    SourceGraph g;
    fscanf(file, "g %d\n", &g.numNodes);
    g.nodes = (SourceNode *)malloc(g.numNodes * sizeof(SourceNode));
    int *indexeForNode = (int *)malloc(g.numNodes * sizeof(int));
    int numEdgesOut = 0, numEdgesIn = 0;
    for (int i = 0; i < g.numNodes; i++)
    {
        fscanf(file, "n %d %d\n", &g.nodes[i].inNeighbours, &g.nodes[i].outNeighbours);
        g.nodes[i].outEdges = (SourceEdge *)malloc(g.nodes[i].outNeighbours * sizeof(SourceEdge));
        indexeForNode[i] = 0;
        numEdgesOut = numEdgesOut + g.nodes[i].outNeighbours;
        numEdgesIn = numEdgesIn + g.nodes[i].inNeighbours;
    }
    printf("Are numEdgesOut == numEdgesIn: %d\n", numEdgesIn == numEdgesOut);
    printf("Total number of edges = %d\n", numEdgesIn);

    for (int i = 0; i < numEdgesIn; i++)
    {
        int source;
        int dest;
        int weight;
        fscanf(file, "e %d %d %d\n", &source, &dest, &weight);
        g.nodes[source].outEdges[indexeForNode[source]].weight = weight;
        g.nodes[source].outEdges[indexeForNode[source]].dest = dest;
        indexeForNode[source]++;
    }

    fclose(file);
    return g;
}
