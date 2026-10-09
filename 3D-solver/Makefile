NVCC      = nvcc
TARGET    = fases3D

# Absolute path to the directory containing this Makefile
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

SRC_DIR   = $(ROOT)core
INC_DIR   = $(ROOT)include
COMMON_DIR= $(ROOT)../common
BUILD_DIR = $(ROOT)build

SRC       = $(SRC_DIR)/main3D.cu
HDRS      = $(SRC_DIR)/functions3D.cuh $(INC_DIR)/parsing3D.h $(INC_DIR)/raytrace3D.h $(INC_DIR)/ism3D.h $(COMMON_DIR)/filterABC.h
OUT       = $(BUILD_DIR)/$(TARGET)

# GPU code: ARCH = this machine's GPU only (fast build). "make release" builds for every GPU
# generation nvcc 11.2 supports (Maxwell 5.2 .. Ampere 8.6) + compute_86 PTX, which the driver
# JIT-compiles on newer GPUs (Ada, Hopper, Blackwell) at first launch.
ARCH         ?= -arch=sm_61
RELEASE_ARCH  = -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 \
                -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
                -gencode arch=compute_86,code=compute_86

# EXTRA lets you append flags without editing this file, e.g. source lines for
# compute-sanitizer:   make EXTRA=-lineinfo
NVCCFLAGS = -std=c++17 -O3 $(ARCH) -I$(INC_DIR) -I$(COMMON_DIR) -Xcompiler "-fopenmp" $(EXTRA)

.PHONY: clean release

$(OUT): $(SRC) $(HDRS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(OUT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(OUT)

# distributable binary: all GPU generations, CUDA runtime linked statically (nvcc default)
release:
	$(MAKE) clean
	$(MAKE) ARCH="$(RELEASE_ARCH)"
