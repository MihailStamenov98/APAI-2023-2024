#include "output_graphs.h"

void printSourceGraph(SourceGraph *g)
{
    for (int i = 0; i < (*g).numNodes; ++i)
    {
        printf("Node %d\n", i);
        printf("Out edges %d\n", g->nodes[i].outNeighbours);
        printf("In edges %d\n", g->nodes[i].inNeighbours);
        for (int j = 0; j < (*g).nodes[i].outNeighbours; ++j)
        {
            printf("Edge source: %d, dest: %d, weight: %d\n", i, (*g).nodes[i].outEdges[j].dest, (*g).nodes[i].outEdges[j].weight);
        }
    }
}

void printDestGraph(DestGraph *g)
{
    for (int i = 0; i < (*g).numNodes; ++i)
    {
        printf("Node %d\n", i);
        for (int j = 0; j < (*g).nodes[i].inNeighbours; ++j)
        {
            printf("Edge source: %d, dest: %d, weight: %d\n", (*g).nodes[i].inEdges[j].source, i, (*g).nodes[i].inEdges[j].weight);
        }
    }
}

void writeGraphToFile(DestGraph *g, const char *filename)
{
    FILE *file = fopen(filename, "wb"); // Open in binary write mode
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    // Write the number of nodes and edges
    fwrite(&g->numNodes, sizeof(int), 1, file);
    fwrite(&g->numEdges, sizeof(int), 1, file);

    // Write the inNeighbours and outNeighbours for each node
    for (int i = 0; i < g->numNodes; i++)
    {
        fwrite(&g->nodes[i].inNeighbours, sizeof(int), 1, file);
        fwrite(&g->nodes[i].outNeighbours, sizeof(int), 1, file);
    }

    // Write the inEdges for each node
    for (int dest = 0; dest < g->numNodes; dest++)
    {
        for (int j = 0; j < g->nodes[dest].inNeighbours; j++)
        {
            // Write the source, destination, and weight of the edge
            fwrite(&g->nodes[dest].inEdges[j].source, sizeof(int), 1, file);
            fwrite(&dest, sizeof(int), 1, file); // Destination node (current node)
            fwrite(&g->nodes[dest].inEdges[j].weight, sizeof(int), 1, file);
        }
    }

    fclose(file);
}
