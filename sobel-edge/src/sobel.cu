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

unsigned char *input_gpu;
unsigned char *output_gpu;

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
                       unsigned char *output,
                       unsigned int height,
                       unsigned int width) {

    char Gx[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    char Gy[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    float valX, valY;
    char _kernal_size = 3;
    int y = blockIdx.y*TILE_SIZE + threadIdx.y;
    int x = blockIdx.x*TILE_SIZE + threadIdx.x;
        valX=0;
        valY=0;
        if ((y>0) && (y<height-1) && (x>0) && (x<width-1)) {
            //calculating the X and Y convolutions
            for (int i = 0; i < _kernal_size; i++) {
                for (int j = 0; j < _kernal_size; j++) {
                    valX += input[i + x - 1 + width * (j + y - 1)] * Gx[i + _kernal_size*j];
                    valY += input[i + x - 1 + width * (j + y - 1)] * Gy[i + _kernal_size*j];
                }
            }
        }
        output[x + y * width] = sqrt(valX*valX + valY*valY);  //Gradient magnitude
}

void SobelEdge(unsigned char *in_mat, 
                   unsigned char *out_mat, 
                   unsigned int height, 
                   unsigned int width){
                         
    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    
    int XSize = gridXSize*TILE_SIZE;
    int YSize = gridYSize*TILE_SIZE;
    
    // Both are the same size (CPU/GPU).
    int size = XSize*YSize;
    
    // Allocate arrays in GPU memory
    checkCuda(cudaMalloc((void**)&input_gpu, size*sizeof(unsigned char)));
    checkCuda(cudaMalloc((void**)&output_gpu, size*sizeof(unsigned char)));
    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
                    
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
    kernel<<<dimGrid, dimBlock>>>(input_gpu, output_gpu, height, width);
    checkCuda(cudaDeviceSynchronize());
    #if defined(CUDA_TIMING)
        TIMER_END(Ktime);
    #endif
    printf("Kernel Execution Time: %f ms\n", Ktime);
        
    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(out_mat, 
                        output_gpu, 
                        height*width*sizeof(unsigned char), 
                        cudaMemcpyDeviceToHost));
                        
    // Free resources and end the program
    checkCuda(cudaFree(output_gpu));
    checkCuda(cudaFree(input_gpu));
}
