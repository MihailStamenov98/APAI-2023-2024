#include "read_graphs.h"

DestGraph *readDestGraphFromFile(const char *filename)
{
    FILE *file = fopen(filename, "rb"); // Open in binary read mode
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    DestGraph *g = (DestGraph *)malloc(sizeof(DestGraph));

    // Read the number of nodes and edges
    fread(&g->numNodes, sizeof(int), 1, file);
    fread(&g->numEdges, sizeof(int), 1, file);

    // Allocate memory for the nodes
    g->nodes = (DestNode *)malloc(g->numNodes * sizeof(DestNode));

    // Read the inNeighbours and outNeighbours for each node
    for (int i = 0; i < g->numNodes; i++)
    {
        fread(&g->nodes[i].inNeighbours, sizeof(int), 1, file);
        fread(&g->nodes[i].outNeighbours, sizeof(int), 1, file);

        // Allocate memory for the inEdges for each node
        g->nodes[i].inEdges = (DestEdge *)malloc(g->nodes[i].inNeighbours * sizeof(DestEdge));
    }

    // Read the inEdges (source, destination, weight) for each node
    for (int dest = 0; dest < g->numNodes; dest++)
    {
        for (int j = 0; j < g->nodes[dest].inNeighbours; j++)
        {
            fread(&g->nodes[dest].inEdges[j].source, sizeof(int), 1, file);
            fread(&dest, sizeof(int), 1, file); // Destination node (this node)
            fread(&g->nodes[dest].inEdges[j].weight, sizeof(int), 1, file);
        }
    }

    fclose(file);
    return g;
}

SourceGraph *readSourceGraphFromFile(const char *filename)
{
    FILE *file = fopen(filename, "rb"); // Open in binary read mode
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    SourceGraph *g = (SourceGraph *)malloc(sizeof(SourceGraph));

    // Read the number of nodes and edges
    fread(&g->numNodes, sizeof(int), 1, file);
    fread(&g->numEdges, sizeof(int), 1, file);

    // Allocate memory for the nodes and index tracking
    g->nodes = (SourceNode *)malloc(g->numNodes * sizeof(SourceNode));
    int *indexForNode = (int *)malloc(g->numNodes * sizeof(int));
    int numEdgesOut = 0, numEdgesIn = 0;

    // Read inNeighbours and outNeighbours for each node
    for (int i = 0; i < g->numNodes; i++)
    {
        fread(&g->nodes[i].inNeighbours, sizeof(int), 1, file);
        fread(&g->nodes[i].outNeighbours, sizeof(int), 1, file);

        // Allocate memory for the outEdges for each node
        g->nodes[i].outEdges = (SourceEdge *)malloc(g->nodes[i].outNeighbours * sizeof(SourceEdge));

        indexForNode[i] = 0;
        numEdgesOut += g->nodes[i].outNeighbours;
        numEdgesIn += g->nodes[i].inNeighbours;
    }

    printf("Are numEdgesOut == numEdgesIn: %d\n", numEdgesIn == numEdgesOut);
    printf("Total number of edges = %d\n", numEdgesIn);

    // Read the edges (source, destination, weight)
    for (int i = 0; i < numEdgesIn; i++)
    {
        int source, dest, weight;
        fread(&source, sizeof(int), 1, file);
        fread(&dest, sizeof(int), 1, file);
        fread(&weight, sizeof(int), 1, file);

        g->nodes[source].outEdges[indexForNode[source]].dest = dest;
        g->nodes[source].outEdges[indexForNode[source]].weight = weight;
        indexForNode[source]++;
    }

    fclose(file);
    return g;
}

#define NO_EDGE_VALUE 21 // Value to represent no edge

int **readGraphToMatrix(const char *filename, int *numNodes)
{
    FILE *file = fopen(filename, "rb"); // Open in binary read mode
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    int numEdges;

    // Read the number of nodes and edges
    fread(numNodes, sizeof(int), 1, file);
    fread(&numEdges, sizeof(int), 1, file);

    // Allocate memory for the adjacency matrix
    int **matrix = (int **)malloc(*numNodes * sizeof(int *));
    for (int i = 0; i < *numNodes; i++)
    {
        matrix[i] = (int *)malloc(*numNodes * sizeof(int));

        // Initialize the matrix to NO_EDGE_VALUE for no edges
        for (int j = 0; j < *numNodes; j++)
        {
            matrix[i][j] = (i == j) ? 0 : NO_EDGE_VALUE; // No self-loops, so 0 for diagonal
        }
    }

    // Read the edges and populate the matrix
    for (int i = 0; i < numEdges; i++)
    {
        int source, dest, weight;

        fread(&source, sizeof(int), 1, file);
        fread(&dest, sizeof(int), 1, file);
        fread(&weight, sizeof(int), 1, file);

        // Populate the matrix with the weight of the edge
        matrix[source][dest] = weight;
    }

    fclose(file);
    return matrix;
}
