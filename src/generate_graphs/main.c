#include "graph_structures.h"
#include "output_graphs.h"
#include "compare_graphs.h"
#include "read_graphs.h"
#include "graph_generator.h"
#include <math.h>
void get_numbers(int index, int *num1, int *num2)
{
    // Calculate the powers of 2 according to index
    *num1 = 512 << ((index + 1) / 2); // First number doubles every 2 indices
    *num2 = 512 << (index / 2);       // Second number doubles every 2 indices starting at index 2
}
int main()
{
    int numnodes, maxNumEdges;
    int edgesCount[18];
    for (int i = 0; i < 9; i++)
    {
        get_numbers(i, &numnodes, &maxNumEdges);
        if (maxNumEdges == numnodes)
        {
            maxNumEdges = maxNumEdges - 1;
        }
        printf("For index = %d, numbers are %d, %d\n", i, numnodes, maxNumEdges);
        DestGraph *destGNoCycle = createGraphNoNegativeCycle(numnodes, maxNumEdges);
        char filenameNoCycle[50];
        snprintf(filenameNoCycle, sizeof(filenameNoCycle), "../../data/graph_no_cycle_%d.edg_%d.txt", numnodes, destGNoCycle->numEdges);
        writeGraphToFile(destGNoCycle, filenameNoCycle);
        freeDestGraph(destGNoCycle);
        edgesCount[2 * i] = destGNoCycle->numEdges;
        DestGraph *destGCycle = createGraphWithNegativeCycle(numnodes, maxNumEdges);
        char filenameCycle[50];
        snprintf(filenameCycle, sizeof(filenameCycle), "../../data/graph_cycle_%d.edg_%d.txt", numnodes, destGCycle->numEdges);
        writeGraphToFile(destGCycle, filenameCycle);
        freeDestGraph(destGCycle);
        edgesCount[2 * i + 1] = destGCycle->numEdges;
    }
    FILE *file = fopen("../../data/stats.txt", "w"); // Open file in write mode
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < 18; i++)
    {
        fprintf(file, "%d\n", edgesCount[i]); // Write each integer to a new line
    }

    fclose(file);
    return 0;
}