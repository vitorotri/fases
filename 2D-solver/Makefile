NVCC      = nvcc
TARGET    = phases2D

# Absolute path to the directory containing this Makefile
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

SRC_DIR   = $(ROOT)core
INC_DIR   = $(ROOT)include
COMMON_DIR= $(ROOT)..//common
BUILD_DIR = $(ROOT)build

SRC       = $(SRC_DIR)/main.cu
HDRS      = $(SRC_DIR)/functions.cuh $(INC_DIR)/parsing.h $(COMMON_DIR)/filterABC.h
OUT       = $(BUILD_DIR)/$(TARGET)

NVCCFLAGS = -std=c++17 -O3 -arch=sm_61 -I$(INC_DIR) -I$(COMMON_DIR)

.PHONY: clean

$(OUT): $(SRC) $(HDRS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(OUT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(OUT)


