NVCC = nvcc
NVCCFLAGS = -std=c++14 -O2 -arch=sm_75
INCLUDE = -I./src -I/usr/include/gtest
LIBS = -lgtest -lgtest_main

SRC_DIR = src/linear_algebra
BUILD_DIR = build
SRC_BUILD_DIR = $(BUILD_DIR)/src/linear_algebra
TEST_DIR = tests

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(SRC_BUILD_DIR)/%.o,$(SRCS))

COMPUTE_SANITIZER = /opt/cuda/extras/compute-sanitizer/compute-sanitizer
SANITIZER_LIB = /opt/cuda/extras/compute-sanitizer

all: $(BUILD_DIR)/run_all_tests

$(SRC_BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(SRC_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

$(BUILD_DIR)/run_all_tests: $(OBJS) $(TEST_DIR)/main_test_runner.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/run_all_tests
	$(BUILD_DIR)/run_all_tests

memcheck: $(BUILD_DIR)/run_all_tests
	LD_LIBRARY_PATH=$(SANITIZER_LIB):$$LD_LIBRARY_PATH $(COMPUTE_SANITIZER) --tool memcheck $(BUILD_DIR)/run_all_tests

.PHONY: all clean test memcheck
