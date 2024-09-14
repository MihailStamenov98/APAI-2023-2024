#include "output_graphs.h"

void printSourceGraph(SourceGraph *g)
{
    for (int i = 0; i < (*g).numNodes; ++i)
    {
        printf("Node %d\n", i);
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

// Function to write a graph to a file
void writeGraphToFile(DestGraph *g, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "g %d %d\n", g->numNodes, g->numEdges);
    for (int i = 0; i < (*g).numNodes; i++)
    {
        fprintf(file, "n %d %d\n", (*g).nodes[i].inNeighbours, (*g).nodes[i].outNeighbours);
    }
    for (int dest = 0; dest < (*g).numNodes; dest++)
    {
        for (int j = 0; j < (*g).nodes[dest].inNeighbours; j++)
        {
            fprintf(file, "e %d %d %d\n", (*g).nodes[dest].inEdges[j].source, dest, (*g).nodes[dest].inEdges[j].weight);
        }
    }

    fclose(file);
}
