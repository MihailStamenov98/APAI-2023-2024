#include <cuda_runtime.h>

#include "cuda_utils.h"

void copySourceGraphToDevice(SourceGraph *h_graph, SourceGraph **d_graph) {
    // Step 1: Allocate memory on the device for the SourceGraph structure
    cudaMalloc((void **)d_graph, sizeof(SourceGraph));

    // Step 2: Allocate memory for the array of SourceNode structures on the device
    SourceNode *d_nodes;
    cudaMalloc((void **)&d_nodes, (*h_graph).numNodes * sizeof(SourceNode));

    // Step 3: Copy each SourceNode, including its outEdges array
    for (int i = 0; i < (*h_graph).numNodes; ++i) {
        SourceNode h_node = (*h_graph).nodes[i];  // Current host node

        // Allocate memory on the device for the outEdges array
        SourceEdge *d_outEdges;
        cudaMalloc((void **)&d_outEdges, h_node.outNeighbours * sizeof(SourceEdge));

        // Copy the outEdges array from the host to the device
        cudaMemcpy(d_outEdges, h_node.outEdges, h_node.outNeighbours * sizeof(SourceEdge), cudaMemcpyHostToDevice);

        // Update the host node to point to the device outEdges
        h_node.outEdges = d_outEdges;

        // Copy the updated node to the device
        cudaMemcpy(&d_nodes[i], &h_node, sizeof(SourceNode), cudaMemcpyHostToDevice);
    }

    // Step 4: Update the host graph to point to the device nodes
    (*h_graph).nodes = d_nodes;

    // Step 5: Copy the updated graph to the device
    cudaMemcpy(*d_graph, &(*h_graph), sizeof(SourceGraph), cudaMemcpyHostToDevice);
}
// Similar implementation for copyDestGraphToDevice()
