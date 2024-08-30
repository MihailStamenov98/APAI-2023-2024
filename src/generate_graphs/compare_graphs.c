#include "compare_graphs.h"

bool compareGraphs(DestGraph g1, SorceGraph g2)
{
    if (g1.numNodes != g2.numNodes)
    {
        printf("g1.numNodes != g2.numNodes\n");
        return false;
    }

    for (int i = 0; i < g1.numNodes; ++i)
    {
        int dest = i;
        if (g1.nodes[i].inNeighbours != g2.nodes[i].inNeighbours)
        {
            printf("g1.nodes[%d].inNeighbours != g2.nodes[%d].inNeighbours\n", i, i);
            printf("g1.nodes[%d].inNeighbours = %d\n", i, g1.nodes[i].inNeighbours);
            printf("g2.nodes[%d].inNeighbours = %d\n", i, g2.nodes[i].inNeighbours);
            return false;
        }
        if (g1.nodes[i].outNeighbours != g2.nodes[i].outNeighbours)
        {
            printf("g1.nodes[%d].outNeighbours != g2.nodes[%d].outNeighbours\n", i, i);
            printf("g1.nodes[%d].outNeighbours = %d\n", i, g1.nodes[i].outNeighbours);
            printf("g2.nodes[%d].outNeighbours = %d\n", i, g2.nodes[i].outNeighbours);
            return false;
        }
        for (int j = 0; j < g1.nodes[i].inNeighbours; j++)
        {

            int source = g1.nodes[dest].inEdges[j].source;
            bool found = false;
            for (int h = 0; h < g2.nodes[source].outNeighbours; h++)
            {
                if (g2.nodes[source].outEdges[h].dest == dest)
                {
                    found = true;
                    if ((g1.nodes[dest].inEdges[j].weight != g2.nodes[source].outEdges[h].weight))
                    {
                        printf("Edge dest: %d, sorce: %d hase different weights in 2 graphs\n", dest, source);
                        printf("In dest graph the weight is %d\n", g1.nodes[dest].inEdges[source].weight);
                        printf("In sorce graph the weight is %d\n", g2.nodes[source].outEdges[dest].weight);
                        return false;
                    }
                }
            }
            if (!found)
            {
                printf("The edge with sorce: %d, dest: %d and weight: %d was not found in sorcegraph\n",
                       source,
                       dest,
                       g1.nodes[dest].inEdges[j].weight);
                return false;
            }
        }
    }

    return true;
}

bool compareDestGraphs(DestGraph g1, DestGraph g2)
{
    if (g1.numNodes != g2.numNodes)
    {
        printf("g1.numNodes != g2.numNodes\n");
        return false;
    }

    for (int i = 0; i < g1.numNodes; ++i)
    {
        if (g1.nodes[i].inNeighbours != g2.nodes[i].inNeighbours)
        {
            printf("g1.nodes[%d].inNeighbours != g2.nodes[%d].inNeighbours\n", i, i);
            printf("g1.nodes[%d].inNeighbours = %d\n", i, g1.nodes[i].inNeighbours);
            printf("g2.nodes[%d].inNeighbours = %d\n", i, g2.nodes[i].inNeighbours);
            return false;
        }
        if (g1.nodes[i].outNeighbours != g2.nodes[i].outNeighbours)
        {
            printf("g1.nodes[%d].outNeighbours != g2.nodes[%d].outNeighbours\n", i, i);
            printf("g1.nodes[%d].outNeighbours = %d\n", i, g1.nodes[i].outNeighbours);
            printf("g2.nodes[%d].outNeighbours = %d\n", i, g2.nodes[i].outNeighbours);
            return false;
        }
        for (int j = 0; j < g1.nodes[i].inNeighbours; j++)
        {
            if ((g1.nodes[i].inEdges[j].source != g2.nodes[i].inEdges[j].source))
            {
                printf("g1.nodes[%d].inEdges[%d].dest != g2.nodes[%d].inEdges[%d].dest\n", i, j, i, j);
                printf("g1.nodes[%d].inEdges[%d].dest = %d\n", i, j, g1.nodes[i].inEdges[j].source);
                printf("g2.nodes[%d].inEdges[%d].dest = %d\n", i, j, g2.nodes[i].inEdges[j].source);

                return false;
            }
            if (g1.nodes[i].inEdges[j].weight != g2.nodes[i].inEdges[j].weight)
            {
                printf("g1.nodes[%d].inEdges[%d].weight != g2.nodes[%d].inEdges[%d].weight\n", i, j, i, j);
                printf("g1.nodes[%d].inEdges[%d].weight = %d\n", i, j, g1.nodes[i].inEdges[j].weight);
                printf("g2.nodes[%d].inEdges[%d].weight = %d\n", i, j, g2.nodes[i].inEdges[j].weight);
                return false;
            }
        }
    }

    return true;
}