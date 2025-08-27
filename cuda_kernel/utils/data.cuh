#pragma once
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <list>
#include <vector>

/**
 * @brief 初始化数据
 * @param ptr 数据指针
 * @param size 数组长度
 */
template <typename T>
void initializeData(T *ptr, int size)
{
    for (int i = 0; i < size; i++)
    {
        ptr[i] = (T)(rand() & 0xFF) / 10.f;
    }
}

template <typename T>
void initializeDataSparse(T *ptr, int size)
{
    double rMax = (double)RAND_MAX;

    for (int i = 0; i < size; i++)
    {
        int r = rand();

        if (r % 3 > 0)
        {
            ptr[i] = 0.0f;
        }
        else
        {
            double dr = (double)r;
            ptr[i]    = (T)(dr / rMax) * 100.0;
        }
    }
}

/**
    @brief 检查运算结果

    @param hostRef 主机端运算结果
    @param gpuRef  设备端运算结果
    @param N       数组长度
    @param rtol    相对误差参数
    @param atol    绝对误差参数
    @param level   Debug信息级别，0为简略信息，1为详细信息，2为全部详细信息

    @details 如果满足不等式abs(a - b) > (atol + rtol * abs(b))则认为不匹配
*/
template <typename T>
void checkResult(T *hostRef, T *gpuRef, const int N, int level = 0, double rtol = 1e-5, double atol = 1e-8)
{
    std::vector<std::pair<int, std::pair<T, T>>> differents;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > (atol + rtol * abs(gpuRef[i])))
        {
            differents.push_back({i, {hostRef[i], gpuRef[i]}});
        }
    }
    int dismatchCount = differents.size();
    if (dismatchCount != 0)
    {
        switch (level)
        {
            case 0:
                printf("Arrays do not match!\n\tindexes: ");
                if (dismatchCount < 10)
                {
                    for (int i = 0; i < dismatchCount; i++)
                        printf("%d ", differents[i].first);
                    printf("\n");
                }
                else
                {
                    for (int i = 0; i < 7; i++)
                        printf("%d ", differents[i].first);
                    printf("... ");
                    for (int i = dismatchCount - 3; i < dismatchCount; i++)
                        printf("%d ", differents[i].first);
                    printf("\ttotal: %d\n", dismatchCount);
                }
                break;
            case 1:
                printf("Arrays do not match!\ttotal: %d\n", dismatchCount);
                if (dismatchCount < 10)
                {
                    std::cout << "+--------------------+--------------------+--------------------+\n";
                    std::cout << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "Index"
                              << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "Host"
                              << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "GPU"
                              << "|\n";
                    std::cout << "+--------------------+--------------------+--------------------+\n";
                    for (int i = 0; i < dismatchCount; i++)
                    {
                        std::cout << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].first
                                  << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].second.first
                                  << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].second.second << "|\n";
                    }
                    std::cout << "+--------------------+--------------------+--------------------+\n";
                }
                else
                {
                    std::cout << "+--------------------+--------------------+--------------------+\n";
                    std::cout << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "Index"
                              << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "Host"
                              << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "GPU"
                              << "|\n";
                    std::cout << "+--------------------+--------------------+--------------------+\n";
                    for (int i = 0; i < 7; i++)
                    {
                        std::cout << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].first
                                  << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].second.first
                                  << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].second.second << "|\n";
                    }
                    std::cout << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "..."
                              << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "..."
                              << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "..."
                              << "|\n";
                    for (int i = dismatchCount - 3; i < dismatchCount; i++)
                    {
                        std::cout << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].first
                                  << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].second.first
                                  << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].second.second << "|\n";
                    }
                    std::cout << "+--------------------+--------------------+--------------------+\n";
                }
                break;
            case 2:
                printf("Arrays do not match!\ttotal: %d\n", dismatchCount);

                std::cout << "+--------------------+--------------------+--------------------+\n";
                std::cout << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "Index"
                          << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "Host"
                          << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << "GPU"
                          << "|\n";
                std::cout << "+--------------------+--------------------+--------------------+\n";
                for (int i = 0; i < dismatchCount; i++)
                {
                    std::cout << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].first
                              << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].second.first
                              << "|" << std::setw(20) << std::setiosflags(std::ios::right) << std::setfill(' ') << differents[i].second.second << "|\n";
                }
                std::cout << "+--------------------+--------------------+--------------------+\n";
                break;
        }
    }
}
