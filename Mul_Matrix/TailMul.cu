
#include <iostream>


#define TILE_SIZE 16


__global__ void TailMul(const float* A, const float* B, float* C, int N){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];  

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    int num_of_tiles = N / TILE_SIZE;
    for(int t = 0; t < num_of_tiles; ++t){
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE+threadIdx.y)*N + col]; 

        __syncthreads();

        // for(int k = 0 ; k< TILE_SIZE; k++){
        //     value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        //     __syncthreads();

        // }
        
        value += As[threadIdx.y][0] * Bs[0][threadIdx.x];
        value += As[threadIdx.y][1] * Bs[1][threadIdx.x];
        value += As[threadIdx.y][2] * Bs[2][threadIdx.x];
        value += As[threadIdx.y][3] * Bs[3][threadIdx.x];
        value += As[threadIdx.y][4] * Bs[4][threadIdx.x];
        value += As[threadIdx.y][5] * Bs[5][threadIdx.x];
        value += As[threadIdx.y][6] * Bs[6][threadIdx.x];
        value += As[threadIdx.y][7] * Bs[7][threadIdx.x];
        value += As[threadIdx.y][8] * Bs[8][threadIdx.x];
        value += As[threadIdx.y][9] * Bs[9][threadIdx.x];
        value += As[threadIdx.y][10] * Bs[10][threadIdx.x];
        value += As[threadIdx.y][11] * Bs[11][threadIdx.x];
        value += As[threadIdx.y][12] * Bs[12][threadIdx.x];
        value += As[threadIdx.y][13] * Bs[13][threadIdx.x];
        value += As[threadIdx.y][14] * Bs[14][threadIdx.x];
        value += As[threadIdx.y][15] * Bs[15][threadIdx.x];
        // __syncthreads();

        }
        C[row * N + col] = value;
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
                          << mat1[i * N + j] << " != " << mat2[i * N + j] << "\n";
                return;
            }
        }
    }
    std::cout << "Matrices are equal!" << "\n";
}


int main(){
    const int N = 1 << 9; // 512
    size_t size = static_cast<size_t>(N) * N * sizeof(float);
    
    
    float *h_A, *h_B, *h_C, *cpu_C;
    cudaMallocHost((void**)&h_A, N*N*sizeof(float));
    cudaMallocHost((void**)&h_B, N*N*sizeof(float));
    cudaMallocHost((void**)&h_C, N*N*sizeof(float));
    cudaMallocHost((void**)&cpu_C, N*N*sizeof(float));
    
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N*N*sizeof(float));
    cudaMalloc((void**)&d_B, N*N*sizeof(float));
    cudaMalloc((void**)&d_C, N*N*sizeof(float)); 
    
    
    for(int i = 0; i < N*N; ++i){
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }   

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); 
    dim3 blockDim(TILE_SIZE, TILE_SIZE); // 16x16 threads per block 
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); // enough blocks to cover N x N matrix 32X32

    TailMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
     
    cal_mul_cpu(h_A, h_B, cpu_C, N);    
    compere_matrix(h_C, cpu_C, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(cpu_C);

    return 0;

}


