# Variables
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -I../generate_graphs -fopenmp
NVCCFLAGS = -I../generate_graphs
LDFLAGS = -fopenmp
LIBS = 

# Source files in folder A (C source files)
SRC_SolverDest = output_structures.c \
                 bellman_ford_parallel_dest.c
SRC_SolverSource = output_structures.c \
                   bellman_ford_parallel_source.c

# Source files in folder B (C source files)
SRC_GenerateGraph = ../generate_graphs/compare_graphs.c \
                    ../generate_graphs/graph_structures.c \
                    ../generate_graphs/output_graphs.c \
                    ../generate_graphs/read_graphs.c \
                    ../generate_graphs/graph_generator.c

# CUDA source files
SRC_CUDA = bellman_ford_cuda.cu \
           cuda_utils.cu

# Object directories
OBJDIR_Solvers = obj
OBJDIR_Generators = obj_generators
OBJDIR_CUDA = obj_cuda

# Object files for C source files
OBJ_dest = $(patsubst %.c,$(OBJDIR_Solvers)/%.o,$(SRC_SolverDest))
OBJ_source = $(patsubst %.c,$(OBJDIR_Solvers)/%.o,$(SRC_SolverSource))

# Object files for graph generators
OBJ_Generators = $(patsubst ../generate_graphs/%.c,$(OBJDIR_Generators)/%.o,$(SRC_GenerateGraph))

# Object files for CUDA files
OBJ_CUDA = $(patsubst %.cu,$(OBJDIR_CUDA)/%.o,$(SRC_CUDA))

# Target executables
PROGRAM_DEST = bf_dest
PROGRAM_SOURCE = bf_source
PROGRAM_CUDA = cuda_prog

# Check for operating system
ifeq ($(OS),Windows_NT)
    RM = del /f /q
    MKDIR = if not exist $(OBJDIR_Solvers) mkdir $(OBJDIR_Solvers)
    MKDIR += && if not exist $(OBJDIR_Generators) mkdir $(OBJDIR_Generators)
    MKDIR += && if not exist $(OBJDIR_CUDA) mkdir $(OBJDIR_CUDA)
    RMDIR = rmdir /s /q
else
    RM = rm -f
    MKDIR = mkdir -p $(OBJDIR_Solvers) && mkdir -p $(OBJDIR_Generators) && mkdir -p $(OBJDIR_CUDA)
    RMDIR = rmdir
endif

# Targets to build specific programs
.PHONY: dest source cuda all clean

# Target A builds bf_dest
dest: $(PROGRAM_DEST)

# Target B builds bf_source
source: $(PROGRAM_SOURCE)

# Target for CUDA program
cuda: $(PROGRAM_CUDA)

# Build the executable for bf_dest
$(PROGRAM_DEST): $(OBJ_dest) $(OBJ_Generators)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Build the executable for bf_source
$(PROGRAM_SOURCE): $(OBJ_source) $(OBJ_Generators)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Build the CUDA executable
$(PROGRAM_CUDA): $(OBJ_CUDA) $(OBJ_Generators)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ 

# Compile source files in folder A to object files in OBJDIR_Solvers
$(OBJDIR_Solvers)/%.o: %.c | $(OBJDIR_Solvers)
	$(MKDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile source files in folder B to object files in OBJDIR_Generators
$(OBJDIR_Generators)/%.o: ../generate_graphs/%.c | $(OBJDIR_Generators)
	$(MKDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA files to object files in OBJDIR_CUDA
$(OBJDIR_CUDA)/%.o: %.cu | $(OBJDIR_CUDA)
	$(MKDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Create the object directories if they don't exist
$(OBJDIR_Solvers) $(OBJDIR_Generators) $(OBJDIR_CUDA):
	$(MKDIR)

# Clean up
clean:
ifeq ($(OS),Windows_NT)
	$(RM) $(OBJDIR_Solvers)\*.o $(OBJDIR_Generators)\*.o $(OBJDIR_CUDA)\*.o 2>nul
	$(RM) $(PROGRAM_DEST) $(PROGRAM_SOURCE) $(PROGRAM_CUDA) 2>nul
	$(RMDIR) /q /s $(OBJDIR_Solvers) 2>nul
	$(RMDIR) /q /s $(OBJDIR_Generators) 2>nul
	$(RMDIR) /q /s $(OBJDIR_CUDA) 2>nul
else
	$(RM) $(OBJDIR_Solvers)/*.o $(OBJDIR_Generators)/*.o $(OBJDIR_CUDA)/*.o 2> /dev/null
	$(RM) $(PROGRAM_DEST) $(PROGRAM_SOURCE) $(PROGRAM_CUDA) 2> /dev/null
	$(RMDIR) $(OBJDIR_Solvers) $(OBJDIR_Generators) $(OBJDIR_CUDA) 2> /dev/null || true
endif
