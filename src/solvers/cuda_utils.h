#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "../generate_graphs/graph_structures.h"
#include "output_structure.h"
// Declare the function to copy a SourceGraph from host to device
void copySourceGraphToDevice(SourceGraph *h_graph, SourceGraph **d_graph);
// void copyBFOutputToHost(BFOutput* out, )

#endif  // CUDA_UTILS_H
