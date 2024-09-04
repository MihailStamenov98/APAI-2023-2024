#include "output_structure.h"

void writeResult(BFOutput out, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }
    if (out.hasNegativeCycle)
    {
        fprintf(file, "There is negative cycle in the graph\n");
        fprintf(file, "From this node one can bactrack to find the cycle = %d\n", out.negativeCycleNode);
    }
    else
    {
        fprintf(file, "There is NOT negative cycle in the graph\n");
    }
    fprintf(file, "timeInSeconds = %d\n", out.timeInSeconds);
    fprintf(file, "numberNodes = %d\n", out.numberNodes);
    for (int i = 0; i < out.numberNodes; i++)
    {
        if (out.hasNegativeCycle)
        {
            fprintf(file, "Predcessor is from node %d to node = %d is %d\n", out.numberNodes);
        }
        else
        {
            fprintf(file, "Distance from node %d to node = %d is %d\n", out.numberNodes);
        }
    }
    fclose(file);
}
