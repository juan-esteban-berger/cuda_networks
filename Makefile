CC=nvcc
CFLAGS=-I./src
OBJDIR=./obj
BINDIR=./bin
SRCDIR=./src

# CUDA architecture flag
CUDA_ARCH=-arch=sm_80

# Dependencies
DEPS = $(SRCDIR)/neural_network.h $(SRCDIR)/linear_algebra.h

# Objects for train, preview and predict
TRAIN_OBJ = $(OBJDIR)/train.o $(OBJDIR)/neural_network.o $(OBJDIR)/linear_algebra.o
PREVIEW_OBJ = $(OBJDIR)/preview.o $(OBJDIR)/linear_algebra.o
PREDICT_OBJ = $(OBJDIR)/predict.o $(OBJDIR)/neural_network.o $(OBJDIR)/linear_algebra.o

# Compile object files for train, preview and predict
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(CUDA_ARCH)

# Linking all object files into final executables
$(BINDIR)/train: $(TRAIN_OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(CUDA_ARCH) -lm

$(BINDIR)/preview: $(PREVIEW_OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(CUDA_ARCH) -lm

$(BINDIR)/predict: $(PREDICT_OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(CUDA_ARCH) -lm

# Target to compile all
all: $(BINDIR)/train $(BINDIR)/preview $(BINDIR)/predict

# Target to clean all object files and executables
.PHONY: clean
clean:
	rm -f $(OBJDIR)/*.o $(BINDIR)/train $(BINDIR)/preview $(BINDIR)/predict

# Target to run train
.PHONY: train
train:
	$(BINDIR)/train 

# Target to run preview
.PHONY: preview 
preview:
	$(BINDIR)/preview

# Target to run predict
.PHONY: predict 
predict:
	$(BINDIR)/predict
