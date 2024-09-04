#include "graph_structures.h"
#include "output_graphs.h"
#include "compare_graphs.h"
#include "read_graphs.h"
#include "graph_generator.h"

int main()
{
    int size = 5;

    DestGraph destGNoCycle = createGraphNoNegativeCycle(size, 4);
    char filenameNoCycle[50];
    snprintf(filenameNoCycle, sizeof(filenameNoCycle), "../../data/no_cycle/graph_no_cycle_%d.txt", size);
    writeGraphToFile(destGNoCycle, filenameNoCycle);
    freeDestGraph(destGNoCycle);

    DestGraph destGCycle = createGraphWithNegativeCycle(size, 4);
    char filenameCycle[50];
    snprintf(filenameCycle, sizeof(filenameCycle), "../../data/cycle/graph_cycle_%d.txt", size);
    writeGraphToFile(destGCycle, filenameCycle);
    freeDestGraph(destGCycle);
    // DestGraph readDestG = readDestGraphFromFile(filenameNoCycle);
    // printf("%d\n", compareDestGraphs(destGNoCycle, readDestG));
    // SourceGraph readSourceG = readSourceGraphFromFile(filenameNoCycle);
    //// printSourceGraph(readSourceG);
    // printf("%d\n", compareGraphs(destGNoCycle, readSourceG));
    //
    // freeDestGraph(readDestG);
    // freeDestGraph(destGNoCycle);
    // freeSourceGraph(readSourceG);
    return 0;
}