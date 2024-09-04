# Variables
CC = gcc
CFLAGS = -Wall -Wextra -I../B
LDFLAGS = -L../B
LIBS = 

# Source files in folder A
SRC_A1 = program1.c
SRC_A2 = program2.c

# Source files in folder B
SRC_B = ../generate_graphs/compare_graphs.c \
      ../generate_graphs/output_graphs.c \
      ../generate_graphs/read_graphs.c \
	  ../generate_graphs/graph_generator.c

# Object directory for A
OBJDIR_A = obj

# Object directory for B
OBJDIR_B = obj_b

# Object files for program1 and program2
OBJ_A1 = $(OBJDIR_A)/program1.o
OBJ_A2 = $(OBJDIR_A)/program2.o

# Object files for B
OBJ_B = $(SRC_B:../B/%.c=$(OBJDIR_B)/%.o)

# Target executables
PROGRAM1 = program1.out
PROGRAM2 = program2.out

# Targets to build specific programs
.PHONY: A B all clean

# Target A builds program1.out
A: $(PROGRAM1)

# Target B builds program2.out
B: $(PROGRAM2)

# Build the executable for program1
$(PROGRAM1): $(OBJ_A1) $(OBJ_B)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Build the executable for program2
$(PROGRAM2): $(OBJ_A2) $(OBJ_B)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Compile source files in folder A to object files in OBJDIR_A
$(OBJDIR_A)/%.o: %.c | $(OBJDIR_A)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile source files in folder B to object files in OBJDIR_B
$(OBJDIR_B)/%.o: ../B/%.c | $(OBJDIR_B)
	$(CC) $(CFLAGS) -c $< -o $@

# Create the object directories if they don't exist
$(OBJDIR_A) $(OBJDIR_B):
	mkdir -p $(OBJDIR_A) $(OBJDIR_B)

# Clean up
clean:
	rm -f $(OBJDIR_A)/*.o $(OBJDIR_B)/*.o $(PROGRAM1) $(PROGRAM2)
	rmdir $(OBJDIR_A) $(OBJDIR_B)
