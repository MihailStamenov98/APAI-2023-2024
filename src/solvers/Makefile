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

# Check for operating system
ifeq ($(OS),Windows_NT)
    RM = del /f /q
    MKDIR = if not exist $(OBJDIR_Solvers) mkdir $(OBJDIR_Solvers)
    MKDIR += && if not exist $(OBJDIR_Generators) mkdir $(OBJDIR_Generators)
    RMDIR = rmdir /s /q
else
    RM = rm -f
    MKDIR = mkdir -p $(OBJDIR_Solvers) &&  mkdir -p $(OBJDIR_Generators) 
    RMDIR = rmdir
endif

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
	$(MKDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile source files in folder B to object files in OBJDIR_Generators
$(OBJDIR_Generators)/%.o: ../generate_graphs/%.c | $(OBJDIR_Generators)
	$(MKDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create the object directories if they don't exist
$(OBJDIR_Solvers) $(OBJDIR_Generators):
	$(MKDIR)

# Clean up
clean:
ifeq ($(OS),Windows_NT)
	$(RM) $(OBJDIR_Solvers)\*.o $(OBJDIR_Generators)\*.o 2>nul
	$(RM) $(PROGRAM_DEST) $(PROGRAM_SOURCE) 2>nul
	$(RMDIR) /q /s $(OBJDIR_Solvers) 2>nul
	$(RMDIR) /q /s $(OBJDIR_Generators) 2>nul
else
	$(RM) $(OBJDIR_Solvers)/*.o $(OBJDIR_Generators)/*.o 2> /dev/null
	$(RM) $(PROGRAM_DEST) $(PROGRAM_SOURCE) 2> /dev/null
	$(RMDIR) $(OBJDIR_Solvers) 2> /dev/null || true
	$(RMDIR) $(OBJDIR_Generators) 2> /dev/null || true
endif

