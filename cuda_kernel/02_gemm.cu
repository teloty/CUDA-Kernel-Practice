#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cublas_v2.h>

#include "utils/common.cuh"
#include "utils/data.cuh"

using data_t = float;
// gemm cpu实现
void gemm_cpu(data_t *A, data_t *B, data_t *C, int M, int N, int K){
    for(int m =0; m<M; m++){
        for(int n=0; n<N; n++){
            data_t sum = 0.0f;
            for(int k=0; k<K; k++){
                sum += A[m*K + k] * B[k*N + n];
            }
            C[m*N + n] = sum;
        }
    }
}

// 调用cublas的gemm实现
void gemm_gpu_cublas(data_t *A, data_t *B, data_t *C, int M, int N, int K){
    float alpha = 1;
    float beta = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasSgemm默认为列主序
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
    // C=A*B=(B^T*A^T)^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);

//     cudaDataType_t input_type = CUDA_R_32F;
//     cudaDataType_t output_type = CUDA_R_32F;
// #if __CUDACC_VER_MAJOR__ >= 11
//     cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
// #else
//     cudaDataType_t compute_type = CUDA_R_32F;
// #endif
//     cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
//     cublasGemmEx(
//         handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
//         &alpha, B, input_type, N, A, input_type, K,
//         &beta, C, output_type, N, compute_type, algo);

    cublasDestroy(handle);
}

/*
 * matrix A, B and C: row-major
 * 2 warp
 *                            N
 *                      --|-------|
 *             B          |0      |
 *                       K|0      |
 *                        |0      |
 *                      --|-------|
 *
 *  A          K              N   
 *         --|---|      --|-------|
 *           |000|        |0      |
 *          M|   |      M |       |
 *           |   |        |       |
 *         --|---|        |-------|
 * 
 * 线程直接从全局内存获取数据
*/
__global__ void gemm_gpu_v1(data_t *A, data_t *B, data_t *C, int M, int N, int K){
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // 总的线程偏移
    int offset = bidx * blockDim.x + tidx;
    // 线程处理C矩阵的行数、列数
    int m = offset / N;
    int n = offset % N;
    if(m>=M || n>=N) return ;
    // K方向乘加
    data_t c_sum = 0.0f;
    for(int k=0; k<K; k++){
        // A行 B列
        c_sum += A[m*K + k] * B[k*N + n];
    }
    // 写回
    C[m*N + n] = c_sum;

}

/*
 * matrix A, B and C: row-major
 * 2 warp
 *
 * C_tile: m8n8 (block tile)
 * A_tile: m8k8
 * B_tile: k8n8
 * 
 * warpid: 0, 1, ...
 *                              N
 *                        --|-------|
 *             B            |0      |
 *                         K|...    |
 *                          |0      |
 *                        --|-------|
 *
 *  A          K                N   
 *         --|-----|      --|-------|
 *           |0...0|        |0|1|...|
 *          M|     |      M |       |
 *           |     |        |       |
 *         --|-----|      --|-------|
 * 
 *                              8
 *                        --|-------|
 *             B _tile      |012...7|
 *                         8|89...15|   ↓
 *                          |...    |   
 *                        --|-------|
 *
 * threadid: 0, 1, ...
 *  A           8               8   
 *         --|-----|      --|-------|
 *           |0...7|        |012...7|
 *    ->    8|8..15|      8 |89...15|
 *           |...  |        |...    |
 *         --|-----|      --|-------|
 * 
 * 使用共享内存进行缓存A、B矩阵，每个线程处理一个C矩阵元素
 * K方向循环
*/
__global__ void gemm_gpu_v2(data_t *A, data_t *B, data_t *C, int M, int N, int K){
    __shared__ data_t A_shared[64];
    __shared__ data_t B_shared[64];
    data_t C_local = 0.0f;
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // 当前block处理C矩阵的行数、列数
    int block_n = (N+7)/8;
    int row = bidx / block_n;
    int col = bidx % block_n;
    data_t *A_ptr = A + row * 8 * K;
    data_t *B_ptr = B + col * 8;
    // 当前线程处理C矩阵的行数、列数
    int m = row*8 + (tidx / 8);
    int n = col*8 + (tidx % 8);
    // 线程的A、B矩阵的偏移
    data_t *A_ptr_local = A_ptr + (tidx/8)*K + (tidx % 8);
    data_t *B_ptr_local = B_ptr + (tidx/8)*N + (tidx % 8);
    // k split loop
    for(int i=0; i<K; i+=8){
        // ldg, 每个线程读取全局内存后存入共享内存
        A_shared[tidx] = A_ptr_local[i];
        B_shared[tidx] = B_ptr_local[i*N];
        __syncthreads();
        // k方向乘加
        for(int j=0; j<8; j++){
            C_local += A_shared[(tidx/8)*8 + j]*B_shared[j*8 + (tidx%8)];
        }
        __syncthreads();

    }
    // 写回
    C[m*N + n] = C_local;
}

/*
 * matrix A, B and C: row-major
 * 2 warp
 *
 * C_tile: m32n8 (block在M方向处理连续4个m8n8) -> m8n8 (block tile)
 * A_tile: m32k8
 * B_tile: k8n8
 * 
 * warpid: 0, 1, ...
 *                              N
 *                        --|-------|
 *             B            |0      |
 *                         K|...    |
 *                          |0      |
 *                        --|-------|
 *
 *  A           K               N   
 *         --|-----|      --|-------|
 *           |0...0|        |0|1|...|
 *           |0...0|        |0|1|...|
 *           |0...0|        |0|1|...|
 *          M|0...0|       M|0|1|...|
 *           |     |        |       |
 *           |     |        |       |
 *         --|-----|      --|-------|
 * 
 * threadid: 0, 1, 2, ...
 *                              8
 *                        --|-------|
 *             B _tile      |012...7|
 *                         8|89...15|   ↓
 *                          |...    |   
 *                        --|-------|
 *
 *  A           8               8   
 *         --|-----|      --|-------|
 *           |0...7|        |012...7|
 *    ->    8|...  |      8 |...    |
 *         --|-----|        |-------|
 *           |0...7|        |012...7|
 *    ->    8|...  |      8 |...    |
 *         --|-----|        |-------|
 *           |0...7|        |012...7|
 *    ->    8|...  |      8 |...    |
 *         --|-----|        |-------|
 *           |0...7|        |012...7|
 *    ->    8|...  |      8 |...    |
 *         --|-----|      --|-------|
 * 
 * 使用共享内存进行缓存A、B矩阵，每个线程处理4个C矩阵元素
 * K方向循环，每次循环读取8个数据
*/
__global__ void gemm_gpu_v3(data_t *A, data_t *B, data_t *C, int M, int N, int K){
    __shared__ data_t A_shared[8*32];
    __shared__ data_t B_shared[8*8];
    // 一个线程负责4个C矩阵元素
    data_t C_local[4];
    for(int i=0; i<4; i++){
        C_local[i] = 0.0f;
    }
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // N方向的block数
    int block_n = (N+7)/8;
    // 当前block处理C矩阵的起始行列数
    int m = bidx / block_n * 32;
    int n = (bidx % block_n) * 8;
    // 当前线程在Block内的行列偏移
    int thread_m = tidx / 8;
    int thread_n = tidx % 8;
    // A、B、C矩阵的指针的内存偏移
    data_t *A_ptr = A + m * K + thread_m * K + (tidx%8);
    data_t *B_ptr = B + (tidx/8)*N + n + thread_n;
    data_t *C_ptr = C + m*N + thread_m*N + n + thread_n;
    // k split loop
    for(int k=0; k<K; k+=8){
        // 当前循环的A，B指针偏移
        data_t *A_ptr_local = A_ptr + k;
        data_t *B_ptr_local = B_ptr + k * N;
        // ldg, 每个线程读取全局内存后存入共享内存
        // A矩阵每个线程读4个元素
        for(int i=0; i<4; i++){
            A_shared[tidx + i*64] = A_ptr_local[i*8*K];
        }
        // B矩阵每个线程读1个元素
        B_shared[tidx] = B_ptr_local[0];
        __syncthreads();

        // k方向乘加
        for(int j=0; j<8; j++){
            data_t b_tmp = B_shared[j*8 + (tidx%8)];
            for(int i=0; i<4; i++){
                C_local[i] += A_shared[i*8*8 + (tidx/8)*8 + j]*b_tmp;
            }
        }
        __syncthreads();
    }

    // 写回
    for(int i=0; i<4; i++){
        C_ptr[i*8*N] = C_local[i];
    }

}

/*
 * matrix A, B and C: row-major
 * 2 warp
 *
 * C_tile: m32n32 (block在M、N方向处理均连续4个m8n8) -> m8n8 (block tile)
 * A_tile: m32k8
 * B_tile: k8n32
 *                                N
 *                        --|----------|
 *             B            |0000      |
 *                         K|...       |
 *                          |0000      |
 *                        --|----------|
 *
 *  A           K                N     
 *         --|-----|      --|----------|
 *           |0...0|        |0000|...  |
 *           |0...0|        |0000|...  |
 *           |0...0|        |0000|...  |
 *          M|0...0|       M|0000|...  |
 *           |     |        |          |
 *           |     |        |          |
 *         --|-----|      --|----------|
 * 
 *                              8       8       8       8
 *                        --|-------|-------|-------|-------|
 *             B _tile      |012...7|012...7|012...7|012...7|
 *                         8|89...15|89...15|89...15|89...15|   ↓
 *                          |...    |...    |...    |...    |    
 *                        --|-------|-------|-------|-------|
 *
 *  A          8                8       8       8       8
 *         --|-----|      --|-------|-------|-------|-------|
 *           |0...7|        |012...7|012...7|012...7|012...7|
 *    ->    8|...  |      8 |...    |...    |...    |...    |
 *         --|-----|        |-------|-------|-------|-------|
 *           |0...7|        |012...7|012...7|012...7|012...7|
 *    ->    8|...  |      8 |...    |...    |...    |...    |
 *         --|-----|        |-------|-------|-------|-------|
 *           |0...7|        |012...7|012...7|012...7|012...7|
 *    ->    8|...  |      8 |...    |...    |...    |...    |
 *         --|-----|        |-------|-------|-------|-------|
 *           |0...7|        |012...7|012...7|012...7|012...7|
 *    ->    8|...  |      8 |...    |...    |...    |...    |
 *         --|-----|      --|-------|-------|-------|-------|
 * 
 * 使用共享内存进行缓存A、B矩阵，每个线程处理16个C矩阵元素
 * K方向循环
*/
__global__ void gemm_gpu_v4(data_t *A, data_t *B, data_t *C, int M ,int N, int K){
    __shared__ data_t A_shared[32*8];
    __shared__ data_t B_shared[8*32];
    data_t C_local[16];
    for(int i=0; i<16; i++){
        C_local[i] = 0.0f;
    }

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // N方向的block数
    int block_n = (N+31)/32;
    // 当前block处理C矩阵的起始行列数
    int m = bidx/block_n * 32;
    int n = (bidx%block_n) * 32;
    // 当前线程在Block内的行列偏移
    int thread_m = tidx/8;
    int thread_n = tidx%8;
    // A、B、C矩阵的指针的内存偏移
    data_t *A_ptr = A + m*K + thread_m*K + (tidx%8);
    data_t *B_ptr = B + (tidx/8)*N + n + thread_n;
    data_t *C_ptr = C + m*N + thread_m*N + n + thread_n;

    // k split loop
    for(int k=0; k<K; k+=8){
        // 当前循环的A，B指针偏移
        data_t *A_ptr_local = A_ptr + k;
        data_t *B_ptr_local = B_ptr + k*N;
        // ldg, 每个线程读取全局内存后存入共享内存
        // A、B矩阵每个线程读4个元素
        for(int i=0; i<4; i++){
            A_shared[tidx + i*64] = A_ptr_local[i*8*K];
            B_shared[tidx + i*64] = B_ptr_local[i*8];
        }
        __syncthreads();

        // k方向乘加
        for(int j=0; j<8; j++){
            for(int a_m_loop=0; a_m_loop<4; a_m_loop++){
                for(int b_n_loop=0; b_n_loop<4; b_n_loop++){
                    C_local[a_m_loop*4 + b_n_loop] += A_shared[a_m_loop*8*8 + (tidx/8)*8 + j]*B_shared[b_n_loop*8*8 + j*8 + (tidx%8)];
                }
            }
        }
        __syncthreads();

    }
    // 写回
    for(int a_m_loop=0; a_m_loop<4; a_m_loop++){
        for(int b_n_loop=0; b_n_loop<4; b_n_loop++){
            C_ptr[a_m_loop*8*N + b_n_loop*8] = C_local[a_m_loop*4 + b_n_loop];
        }
    }

}

/*
 * matrix A, B and C: row-major
 * 2 warp
 *
 * C_tile: m32n32 (block在M、N方向处理均连续4个m8n8) -> m8n8 (block tile)
 * A_tile: m8k32 
 * B_tile: k32n8
 * 每个线程在K方向取4个元素
 *                                N
 *                        --|----------|
 *             B            |0000      |
 *                         K|...       |
 *                          |0000      |
 *                        --|----------|
 *
 *  A           K                N     
 *         --|-----|      --|----------|
 *           |0...0|        |0000|...  |
 *           |0...0|        |0000|...  |
 *           |0...0|        |0000|...  |
 *          M|0...0|       M|0000|...  |
 *           |     |        |          |
 *           |     |        |          |
 *         --|-----|      --|----------|
 * 
 *                                                8       8       8       8
 *                                          --|-------|-------|-------|-------|
 *                                            |0000...|...    |...    |...7777|
 *                                           8|8888...|...    |...    |...    |   ↓
 *                                            |...    |...    |...    |...    |    
 *                                          --|-------|-------|-------|-------|
 *                                            |0000...|...    |...    |...7777|
 *                                           8|8888...|...    |...    |...    |   ↓
 *                                            |...    |...    |...    |...    |    
 *                                          --|-------|-------|-------|-------|
 *                                            |0000...|...    |...    |...7777|
 *                                           8|8888...|...    |...    |...    |   ↓
 *                                            |...    |...    |...    |...    |    
 *                                          --|-------|-------|-------|-------|
 *                                            |0000...|...    |...    |...7777|
 *                                           8|8888...|...    |...    |...    |   ↓
 *                                            |...    |...    |...    |...    |    
 *                                          --|-------|-------|-------|-------|
 *
 * 
 *  A          8      8     8     8               8       8       8       8
 *         --|-----|-----|-----|-----|      --|-------|-------|-------|-------|
 *           |0000.|...  |...  |... 7|        |0000...|...    |...    |...7777|
 *    ->    8|...  |...  |...  |...  |      8 |...    |...    |...    |...    |
 *         --|-----|-----|-----|-----|        |-------|-------|-------|-------|
 *           |0000.|...  |...  |... 7|        |0000...|...    |...    |...7777|
 *    ->    8|...  |...  |...  |...  |      8 |...    |...    |...    |...    |
 *         --|-----|-----|-----|-----|        |-------|-------|-------|-------|
 *           |0000.|...  |...  |... 7|        |0000...|...    |...    |...7777|
 *    ->    8|...  |...  |...  |...  |      8 |...    |...    |...    |...    |
 *         --|-----|-----|-----|-----|        |-------|-------|-------|-------|
 *           |0000.|...  |...  |... 7|        |0000...|...    |...    |...7777|
 *    ->    8|...  |...  |...  |...  |      8 |...    |...    |...    |...    |
 *         --|-----|-----|-----|-----|      --|-------|-------|-------|-------|
 * 
 * 使用共享内存进行缓存A、B矩阵，每个线程处理16个C矩阵元素
 * K方向循环
 * cacheline 128B 32个float，每个线程处理4个float，连续的8个线程处理同一行数据，合并访存开销
*/
__global__ void gemm_gpu_v5(data_t *A, data_t *B, data_t *C, int M, int N, int K){
    __shared__ data_t A_shared[32*32];
    __shared__ data_t B_shared[32*32];
    data_t C_local[4*4];
    for(int i=0; i<16; ++i){
        C_local[i] = 0.0f;
    } 
    // 地址偏移
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // N方向block数
    int block_n = (N + 31)/32;
    // 当前Block的行列偏移
    int m = (bidx/block_n)*32;
    int n = (bidx%block_n)*32;
    // 当前线程在block内的偏移
    int thread_m = tidx/8;
    int thread_n = (tidx%8)*4;
    // A、B、C的全局指针偏移
    data_t *A_ptr = A + m*K + tidx/8*K + (tidx%8)*4; // 一行32个元素，每个线程处理4个元素
    data_t *B_ptr = B + n + (tidx%8)*4 + (tidx/8)*N; // 一行32个元素，每个线程处理4个元素
    data_t *C_ptr = C + m*N + n + thread_m*N + thread_n;
    // K方向循环
    for(int k_loop=0; k_loop<K; k_loop+=32){
        // 当前循环的A，B指针偏移
        data_t *A_ptr_local = A_ptr + k_loop;
        data_t *B_ptr_local = B_ptr + k_loop*N;
        // lds缓存A、B
        for(int i=0; i<4; ++i){
            for(int j=0; j<4; ++j){
                A_shared[i*32*8 + tidx/8*32 + (tidx%8)*4 +j] = A_ptr_local[j + i*8*K];
                B_shared[i*32*8 + tidx/8*32 + (tidx%8)*4 +j] = B_ptr_local[j + i*8*N];
            }
        }
        __syncthreads();

        // 矩阵乘
        for(int k=0; k<32; ++k){
            for(int a_m_loop=0; a_m_loop<4; ++a_m_loop){
                for(int j=0; j<4; ++j){
                    C_local[a_m_loop*4+j] += A_shared[k + a_m_loop*8*32 + (tidx/8)*32]*B_shared[k*32 + (tidx%8)*4 + j];
                    }

                }
            }
        __syncthreads();
    }

    // 写回
    for(int a_m_loop=0; a_m_loop<4; ++a_m_loop){
        for(int j=0; j<4; ++j){
            C_ptr[j + a_m_loop*8*N] = C_local[a_m_loop*4+j];
        }
    }

}

/*
 * float4大位宽读取与写入 for v5
*/
__global__ void gemm_gpu_v6(data_t *A, data_t *B, data_t *C, int M, int N, int K){
    __shared__ data_t A_shared[32*32];
    __shared__ data_t B_shared[32*32];
    data_t C_local[4*4];
    for(int i=0; i<16; ++i){
        C_local[i] = 0.0f;
    } 
    // 地址偏移
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // N方向block数
    int block_n = (N + 31)/32;
    // 当前Block的行列偏移
    int m = (bidx/block_n)*32;
    int n = (bidx%block_n)*32;
    // 当前线程在block内的偏移
    int thread_m = threadIdx.x/8;
    int thread_n = (threadIdx.x%8)*4;
    // A、B、C的全局指针偏移
    data_t *A_ptr = A + m*K + tidx/8*K + (tidx%8)*4; // 一行32个元素，每个线程处理4个元素
    data_t *B_ptr = B + n + (tidx%8)*4 + (tidx/8)*N; // 一行32个元素，每个线程处理4个元素
    data_t *C_ptr = C + m*N + n + thread_m*N + thread_n;
    // K方向循环
    for(int k_loop=0; k_loop<K; k_loop+=32){
        // 当前循环的A，B指针偏移
        data_t *A_ptr_local = A_ptr + k_loop;
        data_t *B_ptr_local = B_ptr + k_loop*N;
        // lds缓存A、B
        for(int i=0; i<4; ++i){
            *(float4*)(A_shared + tidx/8*32 + (tidx%8)*4 + i*32*8) = *(float4*)(A_ptr_local + i*8*K);
            *(float4*)(B_shared + tidx/8*32 + (tidx%8)*4 + i*32*8) = *(float4*)(B_ptr_local + i*8*N);
        }
        __syncthreads();

        // 矩阵乘
        for(int k=0; k<32; ++k){
            for(int a_m_loop=0; a_m_loop<4; ++a_m_loop){
                for(int j=0; j<4; ++j){
                    C_local[a_m_loop*4+j] += A_shared[k + a_m_loop*8*32 + (tidx/8)*32]*B_shared[k*32 + (tidx%8)*4 + j];
                }
            }
        }
        __syncthreads();
    }

    // 写回
    for(int a_m_loop=0; a_m_loop<4; ++a_m_loop){
        *(float4*)(C_ptr + a_m_loop*8*N) = *(float4*)(C_local + a_m_loop*4);
    }

}

/*
* gemmA read
* 0    (01234567)
* 32   (89... 15)
* 64   (16... 23)
* 128  (24... 31)
* 每次gemm的K方向进行shared memory访存时，会发生4次bank conflict
* read bank conflict free for v6
*/
__global__ void gemm_gpu_v7(data_t *A, data_t *B, data_t *C, int M, int N, int K){
    __shared__ data_t A_shared[32*32];
    __shared__ data_t B_shared[32*32];
    data_t C_local[4*4];
    for(int i=0; i<16; ++i){
        C_local[i] = 0.0f;
    }

    // 线程内存储A矩阵的寄存器
    data_t A_load[4*4];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // N方向block数
    int block_n = (N+31)/32;
    // 当前Block的行列偏移
    int m = (bidx/block_n)*32;
    int n = (bidx%block_n)*32;
    // 当前线程在block内的偏移
    int thread_m = tidx/8;
    int thread_n = (tidx%8)*4;
    // A、B、C的全局指针偏移
    data_t *A_ptr = A + m*K + tidx/8*K + (tidx%8)*4; // 一行32个元素，每个线程处理4个元素
    data_t *B_ptr = B + n + (tidx%8)*4 + tidx/8*N; // 一行32个元素，每个线程处理4个元素
    data_t *C_ptr = C + m*N + thread_m*N + n + thread_n;
    // k方向循环
    for(int k_loop=0; k_loop<K; k_loop+=32){
        // 当前循环的A, B指针偏移
        data_t *A_ptr_local = A_ptr + k_loop;
        data_t *B_ptr_local = B_ptr + k_loop*N;
        // load
        for(int i=0; i<4; ++i){
            *(float4*)(A_load + i*4) = *(float4*)(A_ptr_local + i*8*K);
            *(float4*)(B_shared + i*8*32 + tidx/8*32 + (tidx%8)*4) = *(float4*)(B_ptr_local + i*8*N);
        }
        // lds A
        for(int i=0; i<16; ++i){
            int row_write = (tidx/8)%4;
            int col_write = i%4;
            int swizzle_col_write = row_write^col_write;
            // if(tidx<32&&bidx==0){
            //     printf("tid:%d\trow:%d\tcol:%d\tswizzle_col:%d\t\n", tidx, row_write, col_write, swizzle_col_write);
            //     printf("A_shared_offset:%d\trow:%d\tcol:%d\n", i/4*32*8 + tidx/8*32 + (tidx%8)*4 + swizzle_col_write, i/4*8 + tidx/8, (tidx%8)*4 + swizzle_col_write);
            // }
            A_shared[i/4*32*8 + tidx/8*32 + (tidx%8)*4 + swizzle_col_write] = A_load[i];
        }
        __syncthreads();
        // 矩阵乘
        for(int k=0; k<32; ++k){
            int row_read = (tidx/8)%4;
            int col_read = k%4;
            int swizzle_col_read = row_read^col_read;
            for(int a_m_loop=0; a_m_loop<4; ++a_m_loop){
                for(int j=0; j<4; ++j){
                    // if(tidx<32&&bidx==0){
                    //     printf("tid:%d\trow:%d\tcol:%d\tswizzle_col:%d\t\n", tidx, row_read, col_read, swizzle_col_read);
                    //     printf("A_shared_offset:%d\trow:%d\tcol:%d\n", a_m_loop*8*32 + tidx/8*32 + (k/4)*4 + swizzle_col_read, a_m_loop*8 + tidx/8, (k%4)*4 + swizzle_col_read);
                    // }
                    C_local[a_m_loop*4 + j] += A_shared[a_m_loop*8*32 + tidx/8*32 + (k/4)*4 + swizzle_col_read]*B_shared[k*32 + (tidx%8)*4 + j];
                }
            }
        }
        __syncthreads();
        // return ;
    }

    // 写回
    for(int a_m_loop=0; a_m_loop<4; ++a_m_loop){
        *(float4*)(C_ptr + a_m_loop*8*N) = *(float4*)(C_local + a_m_loop*4);
    }

}

/*
* double buffer optim v7
*/
__global__ void gemm_gpu_v8(data_t *A, data_t *B, data_t *C, int M, int N, int K){
    __shared__ data_t A_shared[32*32*2];
    __shared__ data_t B_shared[32*32*2];
    data_t C_local[4*4];
    for(int i=0; i<16; ++i){
        C_local[i] = 0.0f;
    }

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // gemm A load register
    data_t A_load[4*4];
    // N方向的block数
    int block_n = (N+31)/32;
    // 当前block的行列偏移
    int m = (bidx/block_n)*32;
    int n = (bidx%block_n)*32;
    // 当前线程在block内的偏移
    int thread_m = tidx/8;
    int thread_n = (tidx%8)*4;
    // A、B、C的全局指针偏移
    data_t *A_ptr = A + m*K + tidx/8*K + (tidx%8)*4; // 一行32个元素，每个线程处理4个元素
    data_t *B_ptr = B + n + (tidx%8)*4 + tidx/8*N; // 一行32个元素，每个线程处理4个元素
    data_t *C_ptr = C + m*N + thread_m*N + n + thread_n;
    // 读写指针指针
    int read_index = 0;
    int write_index = 0;
    // prefench
    for(int i=0; i<4; ++i){
        *(float4*)(A_load + i*4) = *(float4*)(A_ptr + i*8*K);
        *(float4*)(B_shared + i*8*32 + tidx/8*32 + (tidx%8)*4)=*(float4*)(B_ptr + i*8*N);
    }
    // lds A
    for(int i=0; i<16; ++i){
        int row_write = (tidx/8)%4;
        int col_write = i%4;
        int swizzle_col_write = row_write^col_write;
        A_shared[i/4*32*8+ tidx/8*32 + (tidx%8)*4 + swizzle_col_write] = A_load[i];
    }
    // K方向循环
    for(int k_loop=32; k_loop<K; k_loop+=32){
        __syncthreads();
        // 当前循环的A, B指针偏移
        data_t *A_ptr_local = A_ptr + k_loop;
        data_t *B_ptr_local = B_ptr + k_loop*N;
        // 修改写指针
        write_index = 1 - write_index;
        // load
        for(int i=0; i<4; ++i){
            *(float4*)(A_load + i*4) = *(float4*)(A_ptr_local + i*8*K);
            *(float4*)(B_shared + write_index*32*32 + i*8*32 + tidx/8*32 + (tidx%8)*4) = *(float4*)(B_ptr_local + i*8*N);

        }

        // 矩阵乘
        for(int k=0; k<32; ++k){
            for(int a_m_loop=0; a_m_loop<4; ++a_m_loop){
                for(int j=0; j<4; ++j){
                    int row_read = (tidx/8)%4;
                    int col_read = k%4;
                    int swizzle_col_read = row_read^col_read;
                    C_local[a_m_loop*4 + j] += A_shared[read_index*32*32 + a_m_loop*8*32 + tidx/8*32 + (k/4)*4 + swizzle_col_read]*B_shared[read_index*32*32 + k*32 + (tidx%8)*4 + j];
                }
            }
        }
        // 修改读指针
        read_index = 1 - read_index;

        // lds A
        for(int i=0; i<16; ++i){
            int row_write = (tidx/8)%4;
            int col_write = i%4;
            int swizzle_col_write = row_write^col_write;
            A_shared[write_index*32*32 + i/4*8*32 + tidx/8*32 + (tidx%8)*4 + swizzle_col_write] = A_load[i];
        }
    }
    // last tile
    __syncthreads();
    for(int k=0; k<32; ++k){
        for(int a_m_loop=0; a_m_loop<4; ++a_m_loop){
         for(int j=0; j<4; ++j){
                int row_read = (tidx/8)%4;
                int col_read = k%4;
                int swizzle_col_read = row_read^col_read;
                C_local[a_m_loop*4 + j] += A_shared[read_index*32*32 + a_m_loop*8*32 + tidx/8*32 + (k/4)*4 + swizzle_col_read]*B_shared[read_index*32*32 + k*32 + (tidx%8)*4 + j];
            }
        }
    }
    // 写回
    for(int a_m_loop=0; a_m_loop<4; ++a_m_loop){
        *(float4*)(C_ptr + a_m_loop*8*N) = *(float4*)(C_local + a_m_loop*4);
    }
}

int main(){
    const int M = 4096;
    const int N = 4096;
    const int K = 10240;
    const double relative_error = 1e-2;

    data_t *h_A, *h_B, *h_C, *h_C_gpu;
    h_A = (data_t*)malloc(M*K*sizeof(data_t));
    h_B = (data_t*)malloc(K*N*sizeof(data_t));
    h_C = (data_t*)malloc(M*N*sizeof(data_t));
    h_C_gpu = (data_t*)malloc(M*N*sizeof(data_t));
    // init h_A, h_B
    // for(int i=0; i<M; i++){
    //     for(int j=0; j<K; j++){
    //         h_A[i*K +j] = 1.0f;
    //     }
    // }
    initializeData<data_t>(h_A, M*K);

    // for(int i=0; i<K; i++){
    //     for(int j=0; j<N; j++){
    //         h_B[i*N +j] = 1.0f;
    //     }
    // }
    initializeData<data_t>(h_B, K*N);

    memset(h_C, 0.0f, M*N);
    // record
    // struct timeval tp;
    // gettimeofday(&tp, NULL);
    // double start_cpu = (double)(tp.tv_sec * 1.e3) + (double)(tp.tv_usec * 1.e-3);

    // // cpu for cublas eval
    // gemm_cpu(h_A, h_B, h_C, M, N, K);

    // gettimeofday(&tp, NULL);
    // double elapsed_time_cpu = (double)(tp.tv_sec * 1.e3) + (double)(tp.tv_usec * 1.e-3) - start_cpu;
    // printf("cpu reduce\telapsed %g ms\n", elapsed_time_cpu);

    // gpu
    data_t *d_A, *d_B, *d_C, *d_cublas;
    cudaMalloc((void **)&d_A, M*K*sizeof(data_t));
    cudaMalloc((void **)&d_B, K*N*sizeof(data_t));
    cudaMalloc((void **)&d_C, M*N*sizeof(data_t));
    cudaMalloc((void **)&d_cublas, M*N*sizeof(data_t));

    // device memory copy
    cudaMemcpy(d_A, h_A, M*K*sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M*N);
    cudaMemset(d_cublas, 0, M*N);

    // record
    cudaEvent_t start, stop;
    float time, time_cublas;
    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record the start event
    cudaEventRecord(start);

    // cublas api
    gemm_gpu_cublas(d_A, d_B, d_cublas, M, N, K);
    cudaMemcpy(h_C, d_cublas, M*N*sizeof(data_t), cudaMemcpyDeviceToHost);

    // Record the stop event
    cudaEventRecord(stop);
    // Synchronize to make sure kernel has completed
    cudaEventSynchronize(stop);
    // Calculate the elapsed time
    cudaEventElapsedTime(&time_cublas, start, stop);
    printf("cublas gemm\telapsed %g ms\n", time_cublas);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record the start event
    cudaEventRecord(start);
    // luanch kernel
    // v1 naive global memory gemm
    // const int block_size = 64; // 每个block的线程数
    // const int grid_size = (M*N + block_size - 1) / block_size; // 每个grid的block数
    // gemm_gpu_v1<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

    // v2 block tile shared memory gemm
    // const int block_size = 64; // 每个block的线程数
    // const int grid_size = ((M+7)/8)*((N+7)/8); // 每个grid的block数
    // gemm_gpu_v2<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

    // v3 一维Thread Tile gemm block负责C矩阵M方向的4个块，N方向1个块
    // const int block_size = 64; // 每个block的线程数
    // const int grid_size = ((M+31)/32) * ((N+7)/8); // 每个grid的block数
    // gemm_gpu_v3<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

    // v4 二维Thread Tile gemm block负责C矩阵M方向的4个块，N方向4个块
    // const int block_size = 64; // 每个block的线程数
    // const int grid_size = ((M+31)/32) * ((N+31)/32); // 每个grid的block数
    // gemm_gpu_v4<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

    // v5 cacheline 128B
    // const int block_size = 64; // 每个block的线程数
    // const int grid_size = ((M+31)/32) * ((N+31)/32); // 每个grid的block数
    // gemm_gpu_v5<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

    // v6 cacheline 128B FLOAT4 load
    // const int block_size = 64; // 每个block的线程数
    // const int grid_size = ((M+31)/32) * ((N+31)/32); // 每个grid的block数
    // gemm_gpu_v6<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

    // v7 v6+swizzle A bank conflict free
    // const int block_size = 64; // 每个block的线程数
    // const int grid_size = ((M+31)/32) * ((N+31)/32); // 每个grid的block数
    // gemm_gpu_v7<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

    // v8 v7+double buffer
    const int block_size = 64; // 每个block的线程数
    const int grid_size = ((M+31)/32) * ((N+31)/32); // 每个grid的block数
    gemm_gpu_v8<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);

    // copy back
    cudaMemcpy(h_C_gpu, d_C, M*N*sizeof(data_t), cudaMemcpyDeviceToHost);

    // Record the stop event
    cudaEventRecord(stop);
    // Synchronize to make sure kernel has completed
    cudaEventSynchronize(stop);
    // Calculate the elapsed time
    cudaEventElapsedTime(&time, start, stop);
    printf("gpu gemm\telapsed %g ms\n", time);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            // printf("golden:%f, gpu:%f\n", h_C[i*N + j], h_C_gpu[i*N + j]);
            if(fabs(h_C[i*N + j] - h_C_gpu[i*N + j]) / fabs(h_C[i*N + j] + 1e-10) > relative_error){
                printf("error: %f %f\n", h_C[i*N + j], h_C_gpu[i*N + j]);
                break;
            }
        }
        // printf("\n");
    }
    // 释放指针
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_cublas);

    free(h_C_gpu);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}