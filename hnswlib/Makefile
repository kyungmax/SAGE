# Portable convenience Makefile for the vendored hnswlib tree.
# CMake build files are generated under $(BUILD_DIR).

BUILD_DIR ?= build
CMAKE ?= cmake
CMAKE_BUILD_TYPE ?= Release

.PHONY: all build install clean

all: build

build:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	$(CMAKE) --build $(BUILD_DIR)

install:
	$(CMAKE) --install $(BUILD_DIR)

clean:
	@if [ -d "$(BUILD_DIR)" ]; then $(CMAKE) --build $(BUILD_DIR) --target clean; fi
