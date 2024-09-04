# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra

# Object directory
OBJDIR = obj

# Source files
SRC = compare_graphs.c \
      output_graphs.c \
      read_graphs.c \
      graph_generator.c \
      main.c

# Object files (place .o files in the OBJDIR directory)
OBJ = $(patsubst %.c,$(OBJDIR)/%.o,$(SRC))

# Executable name
TARGET = generate_graphs.exe

# Default rule
all: $(TARGET)

# Link object files into the final executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ)

# Compile each .c file into .o object files in the OBJDIR directory
$(OBJDIR)/%.o: %.c
	@if not exist $(OBJDIR) mkdir $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	del /Q $(OBJDIR)\*.o $(TARGET)
	if exist $(OBJDIR) rmdir $(OBJDIR)

.PHONY: clean
