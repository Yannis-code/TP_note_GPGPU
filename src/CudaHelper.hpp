#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

/* Fonction permettant de vérifier si une erreur est survenue lors d'une fonction CUDA
 * @param err: code d'erreur
 * @param file: nom du fichier
 * @param line: numéro de la ligne
 * @param abort: arrêter le programme
 * @note source: https://stackoverflow.com/a/14038590
 */
inline void gpuAssert(cudaError_t err, const char *file, int line, bool abort = true)
{
    if (err != cudaSuccess)
    {
        std::cerr << "Error in " << file << "(" << line << "): Error : " << cudaGetErrorString(err) << std::endl;
        if (abort)
            exit(err);
    }
}