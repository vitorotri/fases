NVCC      = nvcc
TARGET    = phases2D

SRC_DIR   = src
INC_DIR   = include
BUILD_DIR = build

SRC       = $(SRC_DIR)/main.cu
HDRS      = $(SRC_DIR)/functions.cuh $(INC_DIR)/parsing.h
OUT       = $(BUILD_DIR)/$(TARGET)

NVCCFLAGS = -std=c++17 -O3 -arch=sm_61 -I$(INC_DIR)

.PHONY: run clean

$(OUT): $(SRC) $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)

