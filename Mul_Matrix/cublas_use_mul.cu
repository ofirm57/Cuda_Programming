// #include <cuda_runtime.h>
#include <cstdio>
// #include <cstdlib>
// #include <vector>
#include <iostream>
#include <curand.h> // for random number generation
#include <cublas_v2.h> 
#include <chrono> // for timing


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
            if (fabs(mat1[i * N + j] - mat2[i * N + j]) > 1e-2){
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << mat1[i * N + j] << " != " << mat2[i * N + j] << std::endl;
                return;
            }
        }
    }
    std::cout << "Matrices are equal!" << std::endl;
}


int main() {
    const int N = 1 << 10;
    size_t size = static_cast<size_t>(N) * N * sizeof(float);

    float* h_a = static_cast<float*>(malloc(size));
    float* h_b = static_cast<float*>(malloc(size));
    float* h_c = static_cast<float*>(malloc(size));
    float* cpu_c = static_cast<float*>(malloc(size));

    for (int i = 0; i < N * N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // copy host to device (contiguous full matrix)
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto t0 = std::chrono::steady_clock::now();

    // C = alpha * A * B + beta * C
    // signature: cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //          cublasOperation_t transa, cublasOperation_t transb,
    //          int m, int n, int k,
    //          const float *alpha,
    //          const float *A, int lda,
    //          const float *B, int ldb,
    //          const float *beta,
    //          float *C, int ldc)  
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_b, N, d_a, N, &beta, d_c, N);

    cudaDeviceSynchronize();

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "CUBLAS SGEMM Time: " << ms << " ms" << std::endl;

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cal_mul_cpu(h_a, h_b, cpu_c, N);
    compere_matrix(h_c, cpu_c, N);

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(cpu_c);

    return 0;
}