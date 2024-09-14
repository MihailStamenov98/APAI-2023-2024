#include "graph_structures.h"

void freeDestGraph(DestGraph *graph)
{
    if (graph != NULL)
    {
        // Free the edges for each node
        for (int i = 0; i < graph->numNodes; i++)
        {
            if (graph->nodes[i].inEdges != NULL)
            {
                free(graph->nodes[i].inEdges);
            }
        }
        // Free the nodes array
        if (graph->nodes != NULL)
        {
            free(graph->nodes);
        }
        // Finally, free the graph itself
        free(graph);
    }
}

void freeSourceGraph(SourceGraph *graph)
{
    if (graph != NULL)
    {
        // Free the edges for each node
        for (int i = 0; i < graph->numNodes; i++)
        {
            if (graph->nodes[i].outEdges != NULL)
            {
                free(graph->nodes[i].outEdges);
            }
        }
        // Free the nodes array
        if (graph->nodes != NULL)
        {
            free(graph->nodes);
        }
        // Finally, free the graph itself
        free(graph);
    }
}