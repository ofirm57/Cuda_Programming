
#include <iostream>
#include <cuda_runtime.h>

using std::cout;

#define WARP_SIZE  32

__global__ void AddVectors(int* A,int* B, int* C,int Size){
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    
    if(i < Size){
        C[i] = A[i] + B[i]; 
    }
}


void GenVec(int* A, int* B, int Size){
    for(int i=0 ; i<Size; i++){
        A[i] = i;
        B[i] = Size -i; 
    }
}

void test_res(int* d_res,int* h_res, int Size){
    for(int i=0 ; i < Size; i++){
        if(d_res[i]- h_res[i] == 0)
            cout<<"index "<< i << " = 0 \n";
        else
            cout<<"index "<< i << " !!!== 0 \n" ;
    }

}



void cpu_res(int* A, int* B, int* C, int Size){
    for(int i=0 ; i<Size ; i++){
        C[i] = A[i] + B[i];
}
}



int main(){
    
    int* h_a, *h_b, *h_c, *h_c_cpu;
    int* d_a, *d_b, *d_c;
    int N = 1 << 10;
    size_t byte_size = N * sizeof(int);
    
    cudaMallocHost((void**)&h_a, byte_size);
    cudaMallocHost((void**)&h_b, byte_size);
    cudaMallocHost((void**)&h_c, byte_size);
    cudaMallocHost((void**)&h_c_cpu, byte_size);


    cudaMalloc((void**)&d_a, byte_size );
    cudaMalloc((void**)&d_b, byte_size );
    cudaMalloc((void**)&d_c, byte_size );


    GenVec(h_a,h_b, N);

    cudaMemcpy(d_a, h_a, byte_size , cudaMemcpyDeviceToHost);
    cudaMemcpy(d_b, h_b, byte_size , cudaMemcpyDeviceToHost);
    
    int grid_size = (N + WARP_SIZE -1 ) / WARP_SIZE;
    
    AddVectors <<< grid_size, WARP_SIZE >>> (d_a,d_b,d_c, N);


    cudaMemcpy(h_c, d_c, byte_size , cudaMemcpyHostToDevice);

    cpu_res(h_a,h_b,h_c_cpu, N);

    test_res(h_c_cpu, h_c, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_c_cpu);


    return 0;


}

