# Variables
CC = gcc
CFLAGS = -Wall -Wextra -I../generate_graphs -fopenmp
LDFLAGS = -fopenmp
LIBS = 

# Source files in folder A
SRC_SolverDest = output_structures.c \
				 bellman_ford_parallel_dest.c
SRC_SolverSource = output_structures.c \
				   bellman_ford_parallel_source.c

# Source files in folder B
SRC_GenerateGraph = ../generate_graphs/compare_graphs.c \
      				../generate_graphs/output_graphs.c \
      				../generate_graphs/read_graphs.c \
      				../generate_graphs/graph_generator.c

# Object directory for A
OBJDIR_Solvers = obj

# Object directory for B
OBJDIR_Generators = obj_generators

# Object files for program1 and program2
OBJ_dest = $(patsubst %.c,$(OBJDIR_Solvers)/%.o,$(SRC_SolverDest))
OBJ_source = $(patsubst %.c,$(OBJDIR_Solvers)/%.o,$(SRC_SolverSource))

# Object files for B
OBJ_Generators = $(patsubst ../generate_graphs/%.c,$(OBJDIR_Generators)/%.o,$(SRC_GenerateGraph))

# Target executables
PROGRAM_DEST = bf_dest
PROGRAM_SOURCE = bf_source

# Targets to build specific programs
.PHONY: dest source all clean

# Target A builds bf_dest
dest: $(PROGRAM_DEST)

# Target B builds bf_source
source: $(PROGRAM_SOURCE)

# Build the executable for bf_dest
$(PROGRAM_DEST): $(OBJ_dest) $(OBJ_Generators)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Build the executable for bf_source
$(PROGRAM_SOURCE): $(OBJ_source) $(OBJ_Generators)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Compile source files in folder A to object files in OBJDIR_Solvers
$(OBJDIR_Solvers)/%.o: %.c | $(OBJDIR_Solvers)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile source files in folder B to object files in OBJDIR_Generators
$(OBJDIR_Generators)/%.o: ../generate_graphs/%.c | $(OBJDIR_Generators)
	$(CC) $(CFLAGS) -c $< -o $@

# Create the object directories if they don't exist
$(OBJDIR_Solvers) $(OBJDIR_Generators):
	@if not exist $(OBJDIR_Solvers) mkdir $(OBJDIR_Solvers)
	@if not exist $(OBJDIR_Generators) mkdir $(OBJDIR_Generators)

# Clean up
clean:
	-del /f /q $(OBJDIR_Solvers)\*.o $(OBJDIR_Generators)\*.o 2>nul
	-del /f /q bf_dest.exe bf_source.exe 2>nul
	-rmdir /q /s $(OBJDIR_Solvers) 2>nul
	-rmdir /q /s $(OBJDIR_Generators) 2>nul
