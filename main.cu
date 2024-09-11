#include <cuda_runtime.h>
#include <stdio.h>
int getSmt(int x) { return x + 1; }
#define INF 1000000

// CUDA kernel to add 30 to each element
__global__ void addThirty(int *d_arr, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_arr[idx] = 0;
    }
}
__global__ void relax_initial(int *d_dist, int *d_predecessor, bool *d_wasUpdatedLastIter, bool *d_hasChanged, int n,
                              int startNode, int maxVal) {
    int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
    int i = bdim * bid + tid;
    int skip = bdim * gdim;
    for (int k = i; k < n; k += skip) {
        d_predecessor[k] = -1;
        d_hasChanged[k] = false;

        if (k != startNode) {
            d_dist[k] = maxVal;
            d_wasUpdatedLastIter[k] = false;
        } else {
            d_dist[startNode] = 0;
            d_wasUpdatedLastIter[startNode] = true;
        }
    }
    __syncthreads();
}
int main() {
    int size = 5;
    int *h_arr = (int *)malloc(size * sizeof(int));
    int *d_arr;
    cudaMalloc(&d_arr, size * sizeof(int));

    // Copy host array to device
    // cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to add 30 to each element
    int threadsPerBlock = 16;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    addThirty<<<blocksPerGrid, threadsPerBlock>>>(d_arr, size);

    // Copy results back to host
    cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results and call getSmt
    for (int i = 0; i < size; ++i) {
        printf("Element %d (after init): %d\n", i, h_arr[i]);
    }

    // Free device memory
    cudaFree(d_arr);

    int *h_dist = (int *)malloc(size * sizeof(int));
    int *d_dist;
    cudaMalloc(&d_dist, size * sizeof(int));

    int *h_predecessor = (int *)malloc(size * sizeof(int));
    int *d_predecessor;
    cudaMalloc(&d_predecessor, size * sizeof(int));

    bool *h_wasUpdatedLastIter = (bool *)malloc(size * sizeof(bool));
    bool *d_wasUpdatedLastIter;
    cudaMalloc(&d_wasUpdatedLastIter, size * sizeof(bool));

    bool *h_hasChanged = (bool *)malloc(size * sizeof(bool));
    bool *d_hasChanged;
    cudaMalloc(&d_hasChanged, size * sizeof(bool));

    threadsPerBlock = 1024;
    blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    printf("before cuda malloc\n");

    int startNode = 0;
    relax_initial<<<blocksPerGrid, threadsPerBlock>>>(d_dist, d_predecessor, d_wasUpdatedLastIter, d_hasChanged, size,
                                                      startNode, INF);
    cudaDeviceSynchronize();  // wait for kernel to finish
    cudaMemcpy(h_dist, d_dist, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_predecessor, d_predecessor, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_wasUpdatedLastIter, d_wasUpdatedLastIter, size * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hasChanged, d_hasChanged, size * sizeof(bool), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        printf("iteration %d\n", i);
        printf("h_dist %d,    ", h_dist[i]);
        printf("h_predecessor %d,      ", h_predecessor[i]);
        printf("h_wasUpdatedLastIter %d,      ", h_wasUpdatedLastIter[i]);
        printf("h_hasChanged %d,        \n", h_hasChanged[i]);
    }
    return 0;
}
