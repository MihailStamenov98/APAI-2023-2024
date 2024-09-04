# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra

OBJDIR = obj

# Source files
SRC = ./compare_graphs.c \
      ./output_graphs.c \
      ./read_graphs.c \
	  ./graph_generator.c \
	  ./main.c


# Object files
OBJ = $(SRC:.c=$(OBJDIR_B).o)
# Dependency files
DEPS = $(SRC:.c=.d)
# Executable name
TARGET = generate_graphs

# Default rule
all: $(TARGET)

# Link object files into the final executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ)

# Compile each .c file into .o object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: clean
