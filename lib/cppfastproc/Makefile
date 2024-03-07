INFILE   := fast_proc.cpp
MOD_NAME := cppfastproc
BIN_DIR  := build

# clang produces faster code for skylake
# CC := clang
# CXX := clang++

SUFFIX   := $(shell python3-config --extension-suffix)
LIB_NAME := $(MOD_NAME)$(SUFFIX)

LIB_PATH := $(BIN_DIR)/$(LIB_NAME)

BIN_FLAGS := -O3 -march=native -Wall -std=c++11
# BIN_FLAGS := -Ofast -Wall -std=c++11
LIB_FLAGS := -shared -fPIC

PYTHON_INCLUDES := $(shell python3 -m pybind11 --includes)

# Be verbose about the build.
Q ?= @

.PHONY: pymodule binary clean

pymodule: $(INFILE)
	@echo "===> CXX $(INFILE)"
	$(Q)mkdir -p $(BIN_DIR)
	$(Q)$(CXX) $(BIN_FLAGS) $(LIB_FLAGS) $(PYTHON_INCLUDES) $(INFILE) -o $(LIB_PATH)

binary: $(INFILE)
	@echo "===> CXX $(INFILE)"
	$(Q)mkdir -p $(BIN_DIR)
	$(Q)$(CXX) -DNOPYBIND=1 $(BIN_FLAGS) $(INFILE) -o $(BIN_DIR)/$(MOD_NAME)

clean:
	@echo "===> CLEAN"
	$(Q)rm -rf $(BIN_DIR)/*

