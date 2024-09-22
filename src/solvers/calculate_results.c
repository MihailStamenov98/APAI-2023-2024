#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../generate_graphs/read_graphs.h"

#define MAX_FILES 100
#define MAX_LINE_LENGTH 256

void calculate_throughput(int num_graphs, int *edges, double *times, const char *output_file) {
    FILE *output = fopen(output_file, "w");
    if (output == NULL) {
        printf("Error opening throughput results file.\n");
        return;
    }

    fprintf(output, "Throughput Results (graphs with %d edges):\n", edges);
    for (int i = 0; i < num_graphs; i++) {
        double throughput = (double) edges[i] / times[i];
        fprintf(output, "Graph %d: Throughput = %f items/second\n", i, throughput);
    }

    fclose(output);
}

void calculate_weak_scaling_efficiency(int num_graphs, double *times, const char *output_file) {
    FILE *output = fopen(output_file, "w");
    if (output == NULL) {
        printf("Error opening weak scaling efficiency results file.\n");
        return;
    }

    fprintf(output, "Weak Scaling Efficiency Results (base %d edges):\n", base_edges);
    double base_time = times[0]; // assuming the first time is the base case
    for (int i = 0; i < num_graphs; i++) {
        double efficiency = base_time / times[i];
        fprintf(output, "Graph %d: Weak Scaling Efficiency = %f\n", i, efficiency);
    }

    fclose(output);
}

void read_times_from_file(const char *fileTemplate, int numGraphs, int* edges, double *times, bool hasNegativeCycle) {
    char fileName[MAX_LINE_LENGTH];
    char line[MAX_LINE_LENGTH];
    int numnodes, maxNumEdges;

    for (int i = 0; i < numGraphs; i++) {
        get_numbers(i, &numnodes, &maxNumEdges);
        sprintf(fileName, fileTemplate, numnodes, maxNumEdges);
        FILE *file = fopen(fileName, "r");

        if (file == NULL) {
            printf("Error opening file: %s\n", fileName);
            continue;
        }

        // Read lines until the third row
        for (int line_num = 0; fgets(line, sizeof(line), file) != NULL; line_num++) {
            if(hasNegativeCycle) {
                if (line_num == 2) { // Third row
                sscanf(line, "timeInSeconds = %lf.", &times[i]);
                break;
                }
                if (line_num == 3) { // Third row
                sscanf(line, "numberNodes = %d.", &edges[i]);
                break;
                }
            }
            esle {
                if (line_num == 1) { // Third row
                sscanf(line, "timeInSeconds = %lf.", &times[i]);
                break;
                }
                if (line_num == 2) { // Third row
                sscanf(line, "numberNodes = %d.", &edges[i]);
                break;
                }
            }
            
        }

        fclose(file);
    }
}

int main() {

    double times[12];
    int edges[12];

    // Read CUDA results and calculate throughput
    read_times_from_file("../../results/cuda/graph_no_cycle_%d.edg_%d.txt", 12, edges, times, false);
    calculate_throughput(12, edges, times, "throughput_results_no_cycle.txt");
    printf("No cycle edges\n");
    for(int =i=0; i<12; i++){
        printf("%d\n"edges[i]);
    }
    read_times_from_file("../../results/cuda/graph_cycle_%d.edg_%d.txt", 12, edges, times, false);
    calculate_throughput(12, edges, times, "throughput_results_with_cycle.txt");
    printf("No cycle edges\n");

    for(int =i=0; i<12; i++){
        printf("%d\n"edges[i]);
    }
    // Read OpenMP results and calculate weak scaling efficiency
    read_times_from_file("../../results/omp_dest/graph_no_cycle_%d.edg_%d.txt", 9, times, false);
    calculate_weak_scaling_efficiency(12, edges, times, "weak_scaling_efficiency_results_no_cycle.txt");
    read_times_from_file("../../results/omp_dest/graph_cycle_%d.edg_%d.txt", 9, times, false);
    calculate_weak_scaling_efficiency(12, edges, times, "weak_scaling_efficiency_results_with_cycle.txt");
    
    return 0;
}
