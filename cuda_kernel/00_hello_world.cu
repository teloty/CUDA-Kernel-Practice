#include <stdio.h>
#include <cuda.h>

void print_HelloWorld(){
    printf("Hello World!\n");
}

__global__ void kernel(){
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    printf("Hello World! threadIdx.x = %d, blockIdx.x = %d\n", tidx, bidx);
}

int main(){
    // print_HelloWorld();
    kernel<<<2, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}