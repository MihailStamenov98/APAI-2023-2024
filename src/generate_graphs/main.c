#include "graph_structures.h"
#include "output_graphs.h"
#include "compare_graphs.h"
#include "read_graphs.h"
#include "graph_generator.h"

int main()
{

for (int i = 10; i <= 20; i++) {
    int size = (int)pow(2, i);

    DestGraph* destGNoCycle = createGraphNoNegativeCycle(size, size-1);
    char filenameNoCycle[50];
    snprintf(filenameNoCycle, sizeof(filenameNoCycle), "../../data/no_cycle/graph_no_cycle_%d.txt", size);
    writeGraphToFile(destGNoCycle, filenameNoCycle);
    freeDestGraph(destGNoCycle);

    DestGraph* destGCycle = createGraphWithNegativeCycle(size, size-1);
    char filenameCycle[50];
    snprintf(filenameCycle, sizeof(filenameCycle), "../../data/cycle/graph_cycle_%d.txt", size);
    writeGraphToFile(destGCycle, filenameCycle);
    freeDestGraph(destGCycle);
}
    return 0;
}