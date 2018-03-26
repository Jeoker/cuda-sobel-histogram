#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     
  
#define TILE_SIZE 16
#define CUDA_TIMING
#define RANGE 256

unsigned char *input_gpu;
// unsigned char *output_gpu;

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
    #if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    #endif
    return result;
}
                
// GPU kernel and functions
__global__ void kernel(unsigned char *input, 
                       unsigned int width,
                       unsigned int height,                       
                       unsigned int *hist) {

    int y = blockIdx.y*TILE_SIZE + threadIdx.y;
    int x = blockIdx.x*TILE_SIZE + threadIdx.x;
    int value = input[y * width + x];
    if (x < width && y < height) {
        atomicAdd(&hist[value], 1);
        __syncthreads();
    }
}

void GetHist(unsigned char *in_mat, 
             unsigned int height, 
             unsigned int width, 
             unsigned int *hist) {
                         
    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    
    int XSize = gridXSize*TILE_SIZE;
    int YSize = gridYSize*TILE_SIZE;
    
    unsigned int *hist_gpu;
    
    // Both are the same size (CPU/GPU).
    int size = XSize*YSize;
    
    // Allocate arrays in GPU memory
    checkCuda(cudaMalloc((void**)&input_gpu, size*sizeof(unsigned char)));
    checkCuda(cudaMalloc((void**)&hist_gpu, RANGE*sizeof(unsigned int)));
    
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
                        in_mat, 
                        height*width*sizeof(char), 
                        cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());
    
    // Execute algorithm
    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

    #if defined(CUDA_TIMING)
        float Ktime;
        TIMER_CREATE(Ktime);
        TIMER_START(Ktime);
    #endif
    
    // Kernel Call
    kernel<<<dimGrid, dimBlock>>>(input_gpu, width, height, hist_gpu);
    checkCuda(cudaDeviceSynchronize());
    #if defined(CUDA_TIMING)
        TIMER_END(Ktime);
    #endif
    printf("Kernel Execution Time: %f ms\n", Ktime);
    
    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(hist, 
                        hist_gpu, 
                        RANGE*sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost));
                        
    // Free resources and end the program
    checkCuda(cudaFree(input_gpu));
    checkCuda(cudaFree(hist_gpu));
}
