#include "output_structure.h"

void writeResult(BFOutput* out, const char *filename, bool writeAll)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }
    if ((*out).hasNegativeCycle)
    {
        fprintf(file, "There is negative cycle in the graph\n");
        fprintf(file, "From this node one can bactrack to find the cycle = %d\n", (*out).negativeCycleNode);
    }
    else
    {
        fprintf(file, "There is NOT negative cycle in the graph\n");
    }
    fprintf(file, "timeInSeconds = %lf\n", (*out).timeInSeconds);
    fprintf(file, "numberNodes = %d\n", (*out).numberNodes);
    fprintf(file, "numberEdges = %d\n", out->edgesCount);

    if (writeAll)
    {

        for (int i = 0; i < (*out).numberNodes; i++)
        {
            if ((*out).hasNegativeCycle)
            {
                fprintf(file, "Predcessor of node %d is node %d\n", i, (*out).predecessor[i]);
            }
            else
            {
                fprintf(file, "Distance from node %d to node = %d is %d\n", (*out).startNode, i, (*out).dist[i]);
            }
        }
    }
    fclose(file);
}


void freeBFOutput(BFOutput *output)
{
    if (output != NULL)
    {
        if (output->predecessor != NULL)
        {
            free(output->predecessor);
        }
        if (output->dist != NULL)
        {
            free(output->dist);
        }
        free(output);
    }
}

void printResult(BFOutput* out, bool writeAll)
{
    if ((*out).hasNegativeCycle)
    {
        printf("There is negative cycle in the graph\n");
        printf("From this node one can bactrack to find the cycle = %d\n", (*out).negativeCycleNode);
    }
    else
    {
        printf("There is NOT negative cycle in the graph\n");
    }
    printf("timeInSeconds = %lf\n", (*out).timeInSeconds);
    printf("numberNodes = %d\n", (*out).numberNodes);
    printf("numberEdges = %d\n", out->edgesCount);

    if (writeAll)
    {

        for (int i = 0; i < (*out).numberNodes; i++)
        {
            if ((*out).hasNegativeCycle)
            {
                printf("Predcessor of node %d is node %d\n", i, (*out).predecessor[i]);
            }
            else
            {
                printf("Distance from node %d to node = %d is %d\n", (*out).startNode, i, (*out).dist[i]);
            }
        }
    }
}

BFOutput *initBFOutput(int startNode, int size, int edgesCount)
{
    BFOutput *result;
    result = (BFOutput *)malloc(sizeof(BFOutput));
    result->startNode = startNode;
    result->predecessor = (int *)malloc(size * sizeof(int));
    result->dist = (int *)malloc(size * sizeof(int));
    result->negativeCycleNode = -1;
    result->numberNodes = size;
    result->edgesCount = edgesCount;
    return result;
}
