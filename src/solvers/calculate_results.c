#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "../generate_graphs/read_graphs.h"

#define MAX_FILES 100
#define MAX_LINE_LENGTH 256

void calculate_throughput(int num_graphs, int *edges, double *times, const char *output_file) {
    FILE *output = fopen(output_file, "w");
    if (output == NULL) {
        printf("Error opening throughput results file.\n");
        return;
    }

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
        if (maxNumEdges == numnodes)
        {
            maxNumEdges = maxNumEdges - 1;
        }
        sprintf(fileName, fileTemplate, numnodes, maxNumEdges);
        FILE *file = fopen(fileName, "r");

        if (file == NULL) {
            printf("Error opening file: %s\n", fileName);
            continue;
        }

        // Read lines until the third row
        for (int line_num = 0; line_num < 5; line_num++) {
            fgets(line, sizeof(line), file);
            if(hasNegativeCycle) 
            {
                if (line_num == 2) { // Third row
                    sscanf(line, "timeInSeconds = %lf\n", &times[i]);
                }
                if (line_num == 4) { // Third row
                    sscanf(line, "numberEdges = %d\n", &edges[i]);
                    break;
                }
            }
            else 
            {
                if (line_num == 1) { // Third row
                    sscanf(line, "timeInSeconds = %lf\n", &times[i]);
                }
                if (line_num == 3) { // Third row
                    sscanf(line, "numberEdges = %d\n", &edges[i]);
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
    read_times_from_file("../../results/cuda/graph_cycle_%d.edg_%d.txt", 12, edges, times, true);
    calculate_throughput(12, edges, times, "throughput_results_with_cycle.txt");


    // Read OpenMP results and calculate weak scaling efficiency
    read_times_from_file("../../results/omp_dest/graph_no_cycle_%d.edg_%d.txt", 9, edges, times, false);
    calculate_weak_scaling_efficiency(9, times, "weak_scaling_efficiency_results_no_cycle.txt");
    read_times_from_file("../../results/omp_dest/graph_cycle_%d.edg_%d.txt", 9, edges, times, true);
    calculate_weak_scaling_efficiency(9, times, "weak_scaling_efficiency_results_with_cycle.txt");
    
    return 0;
}
