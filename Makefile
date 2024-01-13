CC=gcc
CFLAGS=-I./src
OBJDIR=./obj
BINDIR=./bin
SRCDIR=./src

# Defining the dependencies, list all .h files used by your .c source files
DEPS = $(SRCDIR)/cuda_neural_network.h $(SRCDIR)/cuda_linear_algebra.h

# Defining the objects, list all .o files required to generate the final executable
_OBJ = main.o cuda_neural_network.o cuda_linear_algebra.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

# Compiling the object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Linking all object files into the final executable
$(BINDIR)/main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) -lm

# This target will compile the project
all: $(BINDIR)/main

# This will clean up the object files and the executable
.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(BINDIR)/main

# This will run the main executable
.PHONY: run
run:
	$(BINDIR)/main
