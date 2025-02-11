#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/*
https://stackoverflow.com/a/14038590
*/
inline void gpuAssert(cudaError_t err, const char *file, int line, bool abort=true)
{
    if (err != cudaSuccess) 
    {
        std::cerr << "Error in " << file << "(" << line << "): Error : " << cudaGetErrorString(err) << std::endl;
        if (abort) exit(err);
    }
}