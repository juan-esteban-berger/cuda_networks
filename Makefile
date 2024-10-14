NVCC = nvcc
NVCCFLAGS = -std=c++14 -O2 -arch=sm_75
INCLUDE = -I./src -I/usr/include/gtest
LIBS = -lgtest -lgtest_main

SRC_DIR = src/linear_algebra
BUILD_DIR = build/src/linear_algebra
TEST_DIR = tests

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS))

COMPUTE_SANITIZER = /opt/cuda/extras/compute-sanitizer/compute-sanitizer
SANITIZER_LIB = /opt/cuda/extras/compute-sanitizer

all: test

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

$(BUILD_DIR)/test_matrix_init: $(OBJS) $(BUILD_DIR)/test_matrix_init.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

$(BUILD_DIR)/test_matrix_init.o: $(TEST_DIR)/test_matrix_init.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_matrix_init
	$(BUILD_DIR)/test_matrix_init

memcheck: $(BUILD_DIR)/test_matrix_init
	LD_LIBRARY_PATH=$(SANITIZER_LIB):$$LD_LIBRARY_PATH $(COMPUTE_SANITIZER) --tool memcheck $(BUILD_DIR)/test_matrix_init

.PHONY: all clean test memcheck
