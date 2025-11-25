// MulM.cu
// Tiled CUDA matrix multiplication (A: MxK, B: KxN => C: MxN)
// Build: nvcc -O2 MulM.cu -o MulM
#include <cuda_runtime.h>
// #include <cstdio>
// #include <cstdlib>
// #include <vector>
#include <iostream>
#include <chrono>



#define CHECK_CUDA(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

const int TILE = 16;

// __global__ void matMulTiled(const float*  d_a, const float*  d_b, float*  d_c, int N){
//     __shared__ float tileA[TILE][TILE];
//     __shared__ float tileB[TILE][TILE];
//     int Row = blockIdx.y * TILE + threadIdx.y;
//     int Col = blockIdx.x * TILE + threadIdx.x;
//     float value = 0.0f; 
//     int num_of_tiles = N / TILE;       
//     for (int t = 0; t < num_of_tiles; ++t) {
//         if (Row < N && t * TILE + threadIdx.x < N)
//             tileA[threadIdx.y][threadIdx.x] = d_a[Row * N + t * TILE + threadIdx.x];
//         else
//             tileA[threadIdx.y][threadIdx.x] = 0.0f;
//         if (t * TILE + threadIdx.y < N && Col < N)
//             tileB[threadIdx.y][threadIdx.x] = d_b[(t * TILE + threadIdx.y) * N + Col];
//         else
//             tileB[threadIdx.y][threadIdx.x] = 0.0f;  
//         __syncthreads();

//         for (int k = 0; k < TILE; ++k) {
//             value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
//         }
//         __syncthreads();

//         d_c[Row * N + Col] = value;

//     }
    
    
// _global__ void matMulTiled_loopunrolling(const float*  d_a, const float*  d_b, float*  d_c, int N){
//     __shared__ float tileA[TILE][TILE];
//     __shared__ float tileB[TILE][TILE];
//     int Row = blockIdx.y * TILE + threadIdx.y;
//     int Col = blockIdx.x * TILE + threadIdx.x;
//     float value = 0.0f; 
//     int num_of_tiles = N / TILE;       
//     for (int t = 0; t < num_of_tiles; ++t) {
//         if (Row < N && t * TILE + threadIdx.x < N)
//             tileA[threadIdx.y][threadIdx.x] = d_a[Row * N + t * TILE + threadIdx.x];
//         else
//             tileA[threadIdx.y][threadIdx.x] = 0.0f;
//         if (t * TILE + threadIdx.y < N && Col < N)
//             tileB[threadIdx.y][threadIdx.x] = d_b[(t * TILE + threadIdx.y) * N + Col];
//         else
//             tileB[threadIdx.y][threadIdx.x] = 0.0f;  
//         __syncthreads();

//         #pragma unroll
//         for (int k = 0; k < TILE; ++k) {
//             value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
//         }
//         __syncthreads();

//         d_c[Row * N + Col] = value;

//     }
// }
    
__global__ void mul_naive(const float*  d_a, const float*  d_b, float*  d_c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += d_a[row * N + k] * d_b[k * N + col];
        }
        d_c[row * N + col] = value;
        // d_c[row][col] = value;
    }

}


void cal_mul_cpu(const float* a, const float* b, float* c, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

void compere_matrix(const float* mat1, const float* mat2, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(mat1[i * N + j] - mat2[i * N + j]) > 1e-4) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << mat1[i * N + j] << " != " << mat2[i * N + j] << std::endl;
                return;
            }
        }
    }
    std::cout << "Matrices are equal!" << std::endl;
}



void genMat(float* mat, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mat[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            // mat[i * N + j] = 2.0;
        }
    }
}



void printMat(const float* mat, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << mat[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv)
{
     // Number of columns in A and rows in B
    int N = 512; 
     int TOT = N*N; // Number of columns in B and C

    float *h_a, *h_b, *h_c, *h_c_cpu;
    cudaMallocHost((void**)&h_a, sizeof(float) * TOT);
    cudaMallocHost((void**)&h_b, sizeof(float) * TOT);
    cudaMallocHost((void**)&h_c, sizeof(float) * TOT);
    cudaMallocHost((void**)&h_c_cpu, sizeof(float) * TOT);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * TOT);
    cudaMalloc((void**)&d_b, sizeof(float) * TOT);
    cudaMalloc((void**)&d_c, sizeof(float) * TOT);

    genMat(h_a, N);
    genMat(h_b, N);

    cudaMemcpy(d_a, h_a, sizeof(float) * TOT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * TOT, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE, TILE);
    dim3 gridSize((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);    


    // matMulTiled<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    mul_naive<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, sizeof(float) * TOT, cudaMemcpyDeviceToHost);

    // std::cout << "Matrix A:" << std::endl;
    // printMat(h_a, N);
    // std::cout << "Matrix B:" << std::endl;
    // printMat(h_b, N);
    // std::cout << "Matrix C (Result):" << std::endl;
    // printMat(h_c, N);
    // // test_result(h_c, N, TOT);

    cal_mul_cpu(h_a,h_b,h_c_cpu, N);
    
    compere_matrix(h_c, h_c_cpu, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);  
    cudaFreeHost(h_c_cpu);

    return 0;
}