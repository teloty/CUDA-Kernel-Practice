#pragma once
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include <stdlib.h>

// 错误检查宏函数
#define ERROR_CHECK(call)                                                                           \
    {                                                                                               \
        const cudaError_t error = call;                                                             \
        if (error != cudaSuccess)                                                                   \
        {                                                                                           \
            printf("CUDA error: %s:%d,\n", __FILE__, __LINE__);                                     \
            printf("\tcode: %s, reason: %s\n", cudaGetErrorName(error), cudaGetErrorString(error)); \
        }                                                                                           \
    }

// cuSPARSE错误检查宏函数
#define ERROR_CHECK_CUSPARSE(call)                                                                          \
    {                                                                                                       \
        const cusparseStatus_t error = call;                                                                \
        if (error != CUSPARSE_STATUS_SUCCESS)                                                               \
        {                                                                                                   \
            printf("cuSPARSE error: %s:%d,\n", __FILE__, __LINE__);                                         \
            printf("\tcode: %s, reason: %s\n", cusparseGetErrorName(error), cusparseGetErrorString(error)); \
        }                                                                                                   \
    }

// cuBLAS错误检查宏函数
#define ERROR_CHECK_CUBLAS(call)                                                                          \
    {                                                                                                     \
        const cublasStatus_t error = call;                                                                \
        if (error != CUBLAS_STATUS_SUCCESS)                                                               \
        {                                                                                                 \
            printf("cuBLAS error: %s:%d,\n", __FILE__, __LINE__);                                         \
            printf("\tcode: %s, reason: %s\n", cublasGetStatusName(error), cublasGetStatusString(error)); \
        }                                                                                                 \
    }

// cuFFT错误检查宏函数
#define ERROR_CHECK_CUFFT(call)                                  \
    {                                                            \
        const cufftResult_t error = call;                        \
        if (error != CUFFT_SUCCESS)                              \
        {                                                        \
            printf("cuFFT error: %s:%d,\n", __FILE__, __LINE__); \
            printf("\tcode: %d\n", error);                       \
        }                                                        \
    }

// cuRAND错误检查宏函数
#define ERROR_CHECK_CURAND(call)                                  \
    {                                                             \
        const curandStatus_t error = call;                        \
        if (error != CURAND_STATUS_SUCCESS)                       \
        {                                                         \
            printf("cuRAND error: %s:%d,\n", __FILE__, __LINE__); \
            printf("\tcode: %d\n", error);                        \
        }                                                         \
    }

// 错误检查普通函数
cudaError_t ErrorCheck(cudaError_t error, const char *filename, int lineNumber)
{
    if (error != cudaSuccess)
    {
        printf("Error: %s:%d,\n", filename, lineNumber);
        printf("\tcode: %s, reason: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        return error;
    }
    return error;
}

// 初始化设备
void setDevice(int device = 0)
{
    int deviceCount   = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&deviceCount), __FILE__, __LINE__); // 获取设备数量
    // 该API返回一个cudaError_t的枚举类
    if (error != cudaError_t::cudaSuccess || deviceCount == 0)
    {
        printf("Get device count faild. There is no device in your computer.\n");
        exit(-1);
    }
    else
    {
        printf("Get device count successfully.\n");
        printf("There %s %d device%s in your computer.\n", (deviceCount > 1 ? "are" : "is"), deviceCount, (deviceCount > 1 ? "s" : ""));
    }

    error = ErrorCheck(cudaSetDevice(device), __FILE__, __LINE__); // 设置执行设备代码的目标设备
    if (error != cudaSuccess)
    {
        printf("Fail to set GPU %d for computing.\n", device);
        exit(-1);
    }
    else
    {
        printf("Set GPU %d for computing.\n", device);
    }
}