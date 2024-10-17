NVCC = nvcc
NVCCFLAGS = -std=c++14 -O2 -arch=sm_75
INCLUDE = -I./src -I/usr/include/gtest
LIBS = -lgtest -lgtest_main

SRC_DIR = src
BUILD_DIR = build
SRC_BUILD_DIR = $(BUILD_DIR)/src
TEST_DIR = tests

COMPUTE_SANITIZER = /opt/cuda/extras/compute-sanitizer/compute-sanitizer
SANITIZER_LIB = /opt/cuda/extras/compute-sanitizer

# Source files
LINEAR_ALGEBRA_SRCS = $(wildcard $(SRC_DIR)/linear_algebra/*.cu)
NEURAL_NETWORK_SRCS = $(wildcard $(SRC_DIR)/neural_network/*.cu)
SRCS = $(LINEAR_ALGEBRA_SRCS) $(NEURAL_NETWORK_SRCS)

# Object files
LINEAR_ALGEBRA_OBJS = $(patsubst $(SRC_DIR)/linear_algebra/%.cu,$(SRC_BUILD_DIR)/linear_algebra/%.o,$(LINEAR_ALGEBRA_SRCS))
NEURAL_NETWORK_OBJS = $(patsubst $(SRC_DIR)/neural_network/%.cu,$(SRC_BUILD_DIR)/neural_network/%.o,$(NEURAL_NETWORK_SRCS))
OBJS = $(LINEAR_ALGEBRA_OBJS) $(NEURAL_NETWORK_OBJS)

all: $(BUILD_DIR)/run_all_tests $(BUILD_DIR)/mnist_classifier

$(SRC_BUILD_DIR)/linear_algebra/%.o: $(SRC_DIR)/linear_algebra/%.cu
	@mkdir -p $(SRC_BUILD_DIR)/linear_algebra
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

$(SRC_BUILD_DIR)/neural_network/%.o: $(SRC_DIR)/neural_network/%.cu
	@mkdir -p $(SRC_BUILD_DIR)/neural_network
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@

$(BUILD_DIR)/run_all_tests: $(OBJS) $(TEST_DIR)/main_test_runner.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ $^ $(LIBS)

$(BUILD_DIR)/mnist_classifier: $(OBJS) $(SRC_DIR)/main.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ $^

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/run_all_tests
	$(BUILD_DIR)/run_all_tests

run: $(BUILD_DIR)/mnist_classifier
	$(BUILD_DIR)/mnist_classifier

memcheck: $(BUILD_DIR)/run_all_tests
	LD_LIBRARY_PATH=$(SANITIZER_LIB):$$LD_LIBRARY_PATH $(COMPUTE_SANITIZER) --tool memcheck $(BUILD_DIR)/run_all_tests

.PHONY: all clean test run memcheck
