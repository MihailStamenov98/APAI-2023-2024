#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define MIN_NODES 1024    // 2^10
#define MAX_NODES 1048576 // 2^20

typedef struct
{
    int source;
    int weight;
} DestEdge;
typedef struct
{
    int dest;
    int weight;
} SorceEdge;
typedef struct
{
    int inNeighbours; // count of edges
    int outNeighbours;
    DestEdge *inEdges;
} DestNode;
typedef struct
{
    int inNeighbours; // count of edges
    int outNeighbours;
    SorceEdge *outEdges;
} SorceNode;
typedef struct
{
    int numNodes;
    DestNode *nodes;
} DestGraph;
typedef struct
{
    int numNodes;
    SorceNode *nodes;
} SorceGraph;

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
        g.nodes[i].inNeighbours = randInt(1, 5);
        g.nodes[i].inEdges = (DestEdge *)malloc(g.nodes[i].inNeighbours * sizeof(DestEdge));
        for (int j = 0; j < g.nodes[i].inNeighbours; j++)
        {
            g.nodes[i].inEdges[j].source = randInt(0, numNodes - 1);
            g.nodes[j].outNeighbours++;
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

// Function to write a graph to a file
void writeGraphToFile(DestGraph g, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "g %d\n", g.numNodes);
    for (int i = 0; i < g.numNodes; i++)
    {
        fprintf(file, "n %d %d\n", g.nodes[i].inNeighbours, g.nodes[i].outNeighbours);
    }
    for (int dest = 0; dest < g.numNodes; dest++)
    {
        for (int j = 0; j < g.nodes[dest].inNeighbours; j++)
        {
            fprintf(file, "e %d %d %d\n", g.nodes[dest].inEdges[j].source, dest, g.nodes[dest].inEdges[j].weight);
        }
    }

    fclose(file);
}

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

    for (int i = 0; i < g.numNodes; i++)
    {
        fscanf(file, "n %d %d\n", &g.nodes[i].inNeighbours, &g.nodes[i].outNeighbours);
        g.nodes[i].outEdges = (SorceEdge *)malloc(g.nodes[i].outNeighbours * sizeof(SorceEdge));
        indexeForNode[i] = 0;
    }
    for (int i = 0; i < g.numNodes; i++)
    {
        for (int j = 0; j < g.nodes[i].outNeighbours; j++)
        {
            int sorce;
            int dest;
            int weight;
            fscanf(file, "e %d %d %d\n", &sorce, &dest, &weight);
            g.nodes[sorce].outEdges[indexeForNode[sorce]].weight = weight;
            g.nodes[sorce].outEdges[indexeForNode[sorce]].dest = dest;
            indexeForNode[sorce]++;
        }
    }

    fclose(file);
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

void printSorceGraph(SorceGraph g)
{
    for (int i = 0; i < g.numNodes; ++i)
    {
        printf("Node %d\n", i);
        for (int j = 0; j < g.nodes[i].outNeighbours; ++j)
        {
            printf("Edge sorce: %d, dest: %d, weight: %d\n", i, g.nodes[i].outEdges[j].dest, g.nodes[i].outEdges[j].weight);
        }
    }
}

void printDestGraph(DestGraph g)
{
    for (int i = 0; i < g.numNodes; ++i)
    {
        printf("Node %d\n", i);
        for (int j = 0; j < g.nodes[i].inNeighbours; ++j)
        {
            printf("Edge sorce: %d, dest: %d, weight: %d\n", g.nodes[i].inEdges[j].source, i, g.nodes[i].inEdges[j].weight);
        }
    }
}
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
    printf("%d\n", compareGraphs(destGNoCycle, readSorceG));

    freeDestGraph(readDestG);
    freeDestGraph(destGNoCycle);
    freeSorceGraph(readSorceG);
    return 0;
}