NVCC      = nvcc
TARGET    = phases3D

# Absolute path to the directory containing this Makefile
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

SRC_DIR   = $(ROOT)core
INC_DIR   = $(ROOT)include
COMMON_DIR= $(ROOT)../common
BUILD_DIR = $(ROOT)build

SRC       = $(SRC_DIR)/main3D.cu
HDRS      = $(SRC_DIR)/functions3D.cuh $(INC_DIR)/parsing3D.h $(COMMON_DIR)/filterABC.h
OUT       = $(BUILD_DIR)/$(TARGET)

NVCCFLAGS = -std=c++17 -O3 -arch=sm_61 -I$(INC_DIR) -I$(COMMON_DIR) -Xcompiler "-fopenmp"

.PHONY: clean

$(OUT): $(SRC) $(HDRS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(OUT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(OUT)
