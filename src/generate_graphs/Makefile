# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra

# Object directory
OBJDIR = obj

# Source files
SRC = graph_structures.c \
      compare_graphs.c \
      output_graphs.c \
      read_graphs.c \
      graph_generator.c \
      main.c

# Object files (place .o files in the OBJDIR directory)
OBJ = $(patsubst %.c,$(OBJDIR)/%.o,$(SRC))

# Executable name
TARGET = generate_graphs

# Check for operating system
ifeq ($(OS),Windows_NT)
    RM = del /Q
    MKDIR = if not exist $(OBJDIR) mkdir $(OBJDIR)
    RMDIR = rmdir $(OBJDIR)
else
    RM = rm -f
    MKDIR = mkdir -p $(OBJDIR)
    RMDIR = rmdir $(OBJDIR)
endif

# Default rule
all: $(TARGET)

# Link object files into the final executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) -lm

# Compile each .c file into .o object files in the OBJDIR directory
$(OBJDIR)/%.o: %.c
	$(MKDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	$(RM) $(OBJDIR)/*.o $(TARGET)
	$(RMDIR)

.PHONY: clean
