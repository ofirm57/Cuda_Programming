# Vector Add 

A small CUDA example that performs element-wise addition of two float arrays (vectors) on the GPU and verifies the result on the host.

## Prerequisites
- CUDA Toolkit (nvcc)
- NVIDIA GPU with appropriate driver
- Linux, macOS (with CUDA support) or Windows (WSL recommended)
- C/C++ compiler

## Files
- vecAdd.cu — CUDA implementation and host code
- Makefile (optional) — build convenience

## Build
Simple nvcc command:
```bash
nvcc -O2 -arch=sm_XX vecAdd.cu -o vecAdd   # replace sm_XX with target compute capability, e.g. sm_75
```
Or with a Makefile:
```bash
make
```

## Usage
Run with optional vector length (default example uses 1<<20):
```bash
./vecAdd [N]
# examples
./vecAdd          # uses default N (e.g. 1048576)
./vecAdd 1000000  # use 1,000,000 elements
```
