#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "utils/common.cuh"
#include "utils/data.cuh"

using data_t = float;
// cpu
// void reduce_sum(data_t *data, data_t *sum, data_t size){
//     for(int i=0; i<size; i++){
//         *sum += data[i];
//     }
// }

// C语言递归实现的归约求和
data_t recursiveReduce(data_t *data, int const size)
{
    if (size == 1)
        return data[0];
    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

// gpu naive atomicAdd
__global__ void reduce_sum_v1(data_t *data, data_t *sum, int size){
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int offset = bidx*blockDim.x + tidx;
    if(offset > size) return ;
    // 原子加实现，在第一个元素位置进行累加
    atomicAdd(sum + 0, data[offset]);
}

// gpu reduce global memory
__global__ void reduce_sum_v2(data_t *data, data_t *block_sum, int size){
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int offset = blockDim.x * bidx + tidx;
    if(offset > size) return ;
    // reduce
    // read data:  0 1 2 3 4 5 6 7 blocksize
    // step1    :  0   2   4   6   活跃线程 (tid%2) == 0 
    // step2    :  0       4       有效线程 (tid%4) == 0 
    // step3    :  0               有效线程 (tid%8) == 0 
    for(int s = 1;s < blockDim.x; s *= 2){
        if((tidx%(s*2)==0)){
            data[offset] += data[offset + s];
        }
        __syncthreads();
    }
    // 每个block的0号线程将数据写入到全局内存
    if(tidx==0)
        block_sum[bidx] = data[offset];
}

// gpu reduce global memory 将全局数据指针转换为当前block的局部数据指针
__global__ void reduce_sum_v3(data_t *data, data_t *block_sum, int size){
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int offset = bidx*blockDim.x + tidx;
    if(offset > size) return ;
    // 声明本地指针指向全局内存, block内连续且均可访问
    data_t *r_data = data + bidx*blockDim.x;
    // reduce
    // read data:  0 1 2 3 4 5 6 7 blocksize
    // step1    :  0   2   4   6   活跃线程 (tid%2) == 0 
    // step2    :  0       4       有效线程 (tid%4) == 0 
    // step3    :  0               有效线程 (tid%8) == 0 
    for(int s = 1; s < blockDim.x; s *= 2){
        if((tidx%(s*2))==0){
            r_data[tidx] += r_data[tidx + s];
        }
        __syncthreads();
    }

    // 每个block的0号线程将数据写入到全局内存
    if(tidx == 0){
        block_sum[bidx] = r_data[0];
    }
}

// gpu reduce shared memory 相邻匹配但线程不连续
template <int block_size>
__global__ void reduce_sum_v4(data_t *data, data_t *block_sum, int size){
    __shared__ data_t s_data[block_size];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // 数据偏移
    int offset = bidx*block_size + tidx;
    if(offset > size) return ;
    // ldg
    s_data[tidx] = data[offset];
    __syncthreads();
    // reduce
    // read data:  0 1 2 3 4 5 6 7 blocksize
    // step1    :  0   2   4   6   活跃线程 (tid%2) == 0 
    // step2    :  0       4       有效线程 (tid%4) == 0 
    // step3    :  0               有效线程 (tid%8) == 0 
    for(int s = 1; s < block_size; s *= 2){
        if((tidx % (2*s)) == 0){
            s_data[tidx] += s_data[tidx + s];
        }
        __syncthreads();
    }

    //每个block的0号线程将数据写入到，全局内存
    if (tidx == 0)
        block_sum[bidx] = s_data[0];
}

// gpu reduce shared memory 相邻匹配强制线程连续
template <int block_size>
__global__ void reduce_sum_v5(data_t *data, data_t *block_sum, int size){
    __shared__ data_t s_data[block_size];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // 数据偏移
    int offset = bidx*block_size + tidx;
    if(offset > size) return ;
    // ldg
    s_data[tidx] = data[offset];
    __syncthreads();
    // reduce
    // read data:  0 1 2 3 4 5 6 7
    // step1    :  0   1   2   3   2 * 1 * tidx 过滤有效线程( < block_size)
    // step2    :  0       1       2 * 2 * tidx 过滤有效线程( < block_size)
    // step3    :  0               1 * 4 * tidx 过滤有效线程( < block_size)
    // 跨步访存smem，存在bank conflict
    for(int s=1; s < block_size; s *= 2){
        int index = 2 * s * tidx;
        if(index < block_size){
            s_data[index] += s_data[index + s];
        }
        __syncthreads();
    }

    if(tidx==0){
        block_sum[bidx] = s_data[0];
    }
}

// gpu reduce shared memory 非相邻匹配且线程连续 smem连续地址，bank free
template <int block_size>
__global__ void reduce_sum_v6(data_t *data, data_t *block_sum, int size){
    __shared__ data_t s_data[block_size];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // 数据偏移
    int offset = bidx*block_size + tidx;
    if(offset > size) return ;
    // ldg
    s_data[tidx] = data[offset];
    __syncthreads();
    // reduce
    // read data:  0 1 2 3 4 5 6 7 blocksize
    // step1    :  0 1 2 3         有效线程 < blocksize/2 
    // step2    :  0 1             有效线程 < blocksize/4 
    // step3    :  0               有效线程 < blocksize/8 
    // 连续访存smem，bank free
    for(int s=block_size/2; s > 0; s /= 2){
        if(tidx < s){
            s_data[tidx] += s_data[tidx + s];
        }
        __syncthreads();
    }

    if(tidx==0){
        block_sum[bidx] = s_data[0];
    }
}

// gpu reduce shared memory 非相邻匹配且线程连续 smem连续地址，bank free, 拿两个数据块
template <int block_size>
__global__ void reduce_sum_v7(data_t *data, data_t *block_sum, int size){
    __shared__ data_t s_data[block_size];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // 数据偏移
    int offset = bidx*block_size + tidx;
    if(offset + block_size > size) return ;
    data_t a1 = data[offset];
    data_t a2 = data[offset + block_size];
    s_data[tidx] = a1 + a2;
    __syncthreads();
    // reduce
    // read data: 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
    // lds:       0 1 2 3 4 5 6 7
    // step1:     0 1 2 3                         有效线程 ( < block_size/2)
    // step2:     0 1                             有效线程 ( < block_size/4)
    // step3:     0                               有效线程 ( < block_size/8)
    for(int s=block_size/2; s > 0; s /= 2){
        if(tidx < s){
            s_data[tidx] += s_data[tidx + s];
        }
        __syncthreads();
    }

    if(tidx==0){
        block_sum[bidx] = s_data[0];
    }
}

// gpu reduce shared memory 非相邻匹配且线程连续 smem连续地址，bank free, 拿四个数据块
template <int block_size>
__global__ void reduce_sum_v8(data_t *data, data_t * block_sum, int size){
    __shared__ data_t s_data[block_size];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // 数据偏移
    int offset = bidx*block_size + tidx;
    if(offset + 3*block_size > size) return ;
    data_t a1 = data[offset];
    data_t a2 = data[offset+block_size];
    data_t a3 = data[offset+2*block_size];
    data_t a4 = data[offset+3*block_size];
    s_data[tidx] = a1 + a2 + a3 + a4;
    __syncthreads();
    // reduce
    for(int s=block_size/2; s > 0; s /= 2){
        if(tidx<s){
            s_data[tidx] += s_data[tidx + s];
        }
        __syncthreads();
    }

    if(tidx==0)
        block_sum[bidx] = s_data[0];
}

// gpu reduce shared memory 非相邻匹配且线程连续 smem连续地址，bank free, 拿八个数据块
template <int block_size>
__global__ void reduce_sum_v9(data_t *data, data_t *block_sum, int size){
    __shared__ data_t s_data[block_size];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int offset = bidx*block_size + tidx;
    // lds
    if(offset + 7*block_size > size) return ;
    data_t a1 = data[offset];
    data_t a2 = data[offset+block_size];
    data_t a3 = data[offset+2*block_size];
    data_t a4 = data[offset+3*block_size];
    data_t a5 = data[offset+4*block_size];
    data_t a6 = data[offset+5*block_size];
    data_t a7 = data[offset+6*block_size];
    data_t a8 = data[offset+7*block_size];
    s_data[tidx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    __syncthreads();

    // reduce
    for(int s = block_size/2; s > 0; s /= 2){
        if(tidx<s){
            s_data[tidx] += s_data[tidx + s];
        }
        __syncthreads();
    }

    if(tidx==0)
        block_sum[bidx] = s_data[0];
}

// gpu reduce shared memory 非相邻匹配且线程连续 smem连续地址，bank free, 拿八个数据块 最后一个warp循环展开
// 最后一个warp循环展开 volatile修饰防止编译器进行优化, 每次访问均从内存中读取
__device__ void warpReduce(volatile data_t *s_data, int tidx){
    s_data[tidx] += s_data[tidx + 32];
    s_data[tidx] += s_data[tidx + 16];
    s_data[tidx] += s_data[tidx + 8];
    s_data[tidx] += s_data[tidx + 4];
    s_data[tidx] += s_data[tidx + 2];
    s_data[tidx] += s_data[tidx + 1];
}

template <int block_size>
__global__ void reduce_sum_v10(data_t *data, data_t *block_sum, int size){
    __shared__ data_t s_data[block_size];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int offset = bidx*block_size + tidx;
    // lds
    if(offset + 7*block_size > size) return ;
    data_t a1 = data[offset];
    data_t a2 = data[offset+block_size];
    data_t a3 = data[offset+block_size*2];
    data_t a4 = data[offset+block_size*3];
    data_t a5 = data[offset+block_size*4];
    data_t a6 = data[offset+block_size*5];
    data_t a7 = data[offset+block_size*6];
    data_t a8 = data[offset+block_size*7];
    s_data[tidx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    __syncthreads();

    // reduce
    for(int s=block_size/2; s > 32; s /=2){
        if(tidx<s){
            s_data[tidx] += s_data[tidx + s];
        }
        __syncthreads();
    }
    __syncthreads();
    
    // 最后一个warp循环展开
    if(tidx<32){
        // warpReduce(s_data, tidx);
        // 被 volatile 修饰的变量，每次访问都会从内存中读取，而不是从寄存器中读取。
        // 防止编译器对这些变量进行优化。
        // 如果不加 volatile 修饰符，编译器会认为这些变量的值不会变化，所以会将这些变量的值缓存在寄存器中。 
        // 可能导致读到的值不是最新的值。
        volatile data_t *s_data_v = s_data;
        s_data_v[tidx] += s_data_v[tidx + 32];
        s_data_v[tidx] += s_data_v[tidx + 16];
        s_data_v[tidx] += s_data_v[tidx + 8];
        s_data_v[tidx] += s_data_v[tidx + 4];
        s_data_v[tidx] += s_data_v[tidx + 2];
        s_data_v[tidx] += s_data_v[tidx + 1];

    }

    if(tidx==0){
        block_sum[bidx] = s_data[0];
    }
}

// gpu reduce shared memory 非相邻匹配且线程连续 smem连续地址，bank free, 拿八个数据块 完全循环展开
template <int block_size>
__global__ void reduce_sum_v11(data_t *data, data_t *block_sum, int size){
    __shared__ data_t s_data[block_size];
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int offset = bidx*block_size + tidx;
    // lds
    if(offset + 7*block_size > size) return ;
    data_t a1 = data[offset];
    data_t a2 = data[offset+block_size];
    data_t a3 = data[offset+block_size*2];
    data_t a4 = data[offset+block_size*3];
    data_t a5 = data[offset+block_size*4];
    data_t a6 = data[offset+block_size*5];
    data_t a7 = data[offset+block_size*6];
    data_t a8 = data[offset+block_size*7];
    s_data[tidx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    __syncthreads();
    // reduce 完全展开
    if(block_size >= 1024 && tidx < 512)
        s_data[tidx] += s_data[tidx + 512];
    __syncthreads();

    if(block_size >= 512 && tidx < 256)
        s_data[tidx] += s_data[tidx + 256];
    __syncthreads();

    if(block_size >= 256 && tidx < 128)
        s_data[tidx] += s_data[tidx + 128];
    __syncthreads();

    if(block_size >= 128 && tidx < 64)
        s_data[tidx] += s_data[tidx + 64];
    __syncthreads();

    if(block_size >= 64 && tidx < 32){
        // warpReduce(s_data, tidx);
        volatile data_t *s_data_v = s_data;
        s_data_v[tidx] += s_data_v[tidx + 32];
        s_data_v[tidx] += s_data_v[tidx + 16];
        s_data_v[tidx] += s_data_v[tidx + 8];
        s_data_v[tidx] += s_data_v[tidx + 4];
        s_data_v[tidx] += s_data_v[tidx + 2];
        s_data_v[tidx] += s_data_v[tidx + 1];
    }

    if(tidx==0)
        block_sum[bidx] = s_data[0];
        
}

int main(){
    setDevice();
    // 数据规模大小
    const unsigned int size = 1<<24;
    const int block_size = 64;
    const int grid_size = (size + block_size - 1) / (block_size);
    data_t cpu_sum, gpu_sum;
    cpu_sum = 0.0f, gpu_sum = 0.0f;
    // 分配host端内存
    data_t *h_data, *h_out_data, *temp;
    h_data = (data_t *)malloc(size*sizeof(data_t));
    temp = (data_t *)malloc(size*sizeof(data_t));

    h_out_data = (data_t *)malloc(grid_size*sizeof(data_t));
    // initializeData<data_t>(h_data, size);
    for(int i=0; i<size; i++){
        h_data[i] = 1.0f;
    }
    memcpy(temp, h_data, size*sizeof(data_t));

    struct timeval tp;
    gettimeofday(&tp, NULL);
    double start_cpu = (double)(tp.tv_sec * 1.e3) + (double)(tp.tv_usec * 1.e-3);

    cpu_sum = recursiveReduce(temp, size);

    gettimeofday(&tp, NULL);
    double elapsed_time_cpu = (double)(tp.tv_sec * 1.e3) + (double)(tp.tv_usec * 1.e-3) - start_cpu;
    printf("cpu reduce\telapsed %g ms\tsum of host: %f\n", elapsed_time_cpu, cpu_sum);

    cudaEvent_t start, stop;
    float time;
    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record the start event
    cudaEventRecord(start);

    // 分配device端内存
    data_t *d_data, *block_sum;
    cudaMalloc((void **)&d_data, size*sizeof(data_t));
    cudaMalloc((void **)&block_sum, sizeof(data_t)*grid_size);
    // 将host端数据拷贝到device端
    cudaMemcpy(d_data, h_data, size*sizeof(data_t), cudaMemcpyHostToDevice);

    // 调用kernel函数
    // v1 atomic add
    // data_t *d_sum;
    // cudaMalloc((void **)&d_sum, sizeof(data_t));
    // // 赋0
    // cudaMemset(d_sum, 0, sizeof(data_t));
    // reduce_sum_v1<<<grid_size, block_size>>>(d_data, d_sum, size);
    // // 将device端结果拷贝到host端
    // cudaMemcpy(&gpu_sum, d_sum, sizeof(data_t), cudaMemcpyDeviceToHost);
    // cudaFree(d_sum);

    // v2 global memory
    // reduce_sum_v2<<<grid_size, block_size>>>(d_data, block_sum, size);
    // cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size, cudaMemcpyDeviceToHost);
    // for(int i=0; i<grid_size; i++){
    //     gpu_sum += h_out_data[i];
    // }

    // v2 global memory 将全局数据指针转换为当前block的局部数据指针
    // reduce_sum_v3<<<grid_size, block_size>>>(d_data, block_sum, size);
    // cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size, cudaMemcpyDeviceToHost);
    // for(int i=0; i<grid_size; i++){
    //     gpu_sum += h_out_data[i];
    // }

    // v4 shared memory 相邻匹配但线程不连续
    // reduce_sum_v4<block_size><<<grid_size, block_size>>>(d_data, block_sum, size);
    // cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size, cudaMemcpyDeviceToHost);
    // for(int i=0; i<grid_size; i++){
    //     gpu_sum += h_out_data[i];
    // }

    // v5 shared memory 相邻匹配强制线程连续
    // reduce_sum_v5<block_size><<<grid_size, block_size>>>(d_data, block_sum, size);
    // cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size, cudaMemcpyDeviceToHost);
    // for(int i=0; i<grid_size; i++){
    //     gpu_sum += h_out_data[i];
    // }

    // v6 shared memory 交错匹配，解bank conflict
    // reduce_sum_v6<block_size><<<grid_size, block_size>>>(d_data, block_sum, size);
    // cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size, cudaMemcpyDeviceToHost);
    // for(int i=0; i<grid_size; i++){
    //     gpu_sum += h_out_data[i];
    // }

    // v7 shared memory 交错匹配且线程连续 smem连续地址，bank free, 拿两个数据块
    // reduce_sum_v7<block_size><<<grid_size / 2, block_size>>>(d_data, block_sum, size);
    // cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size/2, cudaMemcpyDeviceToHost);
    // for(int i=0; i<grid_size/2; i++){
    //     gpu_sum += h_out_data[i];
    // }

    // v8 shared memory 交错匹配且线程连续 smem连续地址，bank free, 拿四个数据块
    // reduce_sum_v8<block_size><<<grid_size / 4, block_size>>>(d_data, block_sum, size);
    // cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size/4, cudaMemcpyDeviceToHost);
    // for(int i=0; i<grid_size/4; i++){
    //     gpu_sum += h_out_data[i];
    // }

    // v9 shared memory 交错匹配且线程连续 smem连续地址，bank free, 拿八个数据块
    // reduce_sum_v9<block_size><<<grid_size / 8, block_size>>>(d_data, block_sum, size);
    // cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size/8, cudaMemcpyDeviceToHost);
    // for(int i=0; i<grid_size/8; i++){
    //     gpu_sum += h_out_data[i];
    // }

    // v10 shared memory 交错匹配且线程连续 smem连续地址，bank free, 拿八个数据块，最后一个warp循环展开
    // reduce_sum_v10<block_size><<<grid_size / 8, block_size>>>(d_data, block_sum, size);
    // cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size/8, cudaMemcpyDeviceToHost);
    // for(int i=0; i<grid_size/8; i++){
    //     gpu_sum += h_out_data[i];
    // }

    // *** FINAL VERSION ***
    // v11 shared memory 交错匹配且线程连续 smem连续地址，bank free, 拿八个数据块，完全循环展开
    reduce_sum_v11<block_size><<<grid_size / 8, block_size>>>(d_data, block_sum, size);
    cudaMemcpy(h_out_data, block_sum, sizeof(data_t)*grid_size/8, cudaMemcpyDeviceToHost);
    for(int i=0; i<grid_size/8; i++){
        gpu_sum += h_out_data[i];
    }

    // Record the stop event
    cudaEventRecord(stop);
    // Synchronize to make sure kernel has completed
    cudaEventSynchronize(stop);
    // Calculate the elapsed time
    cudaEventElapsedTime(&time, start, stop);
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("gpu reduce\telapsed %g ms\tsum of device: %f\n", time, gpu_sum);

    // 释放内存
    cudaFree(d_data);
    cudaFree(block_sum);

    free(h_data);
    free(temp);
    free(h_out_data);

    // Reset the device
    ERROR_CHECK(cudaDeviceReset());

    // 检查结果
    if (cpu_sum == gpu_sum){
        printf("Success!\n");
    }else{
        printf("Fail!\n");
    }

    return 0;
}
