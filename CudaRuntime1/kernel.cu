#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define NUM_THREADS 128
#define NUM_CHANNELS 1
// Random sample from the image for debug purposes.
#define DEBUG_IMG_IDX 20
// AdderNET kernel size
#define KERNEL_RADIUS 5

typedef unsigned char uint8_t;

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdint.h>

cudaError_t addernetCUDA(uint8_t* out_img, uint8_t* img, uint8_t* addernet_kernel, const int kernel_radius, const int img_width, const int img_length);

// Assuming VALID padding (size decreases)
// Index: kernel_radius -> img_width - kernel_radius - 1
// using square kernels like 5x5 (radius 2), 11x11 (radius 5)
// Single input & output channels
__global__ void addernetKernel(uint8_t* dev_out, uint8_t* dev_img, uint8_t* addernet_kernel, const int kernel_radius, const int img_width, const int img_length) {

    unsigned int tid_x = threadIdx.x;
    unsigned int tid_y = threadIdx.y;
    // Kernel radius can be added in the loop instead of using minus values.
    unsigned int idx_x = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int idx_y = blockIdx.y * (blockDim.y * 2) + threadIdx.y;

    int accumulator = 0;
    int kernel_length = 2 * kernel_radius + 1; 

    for (size_t j = 0; j <= kernel_length; j++)
    {
        for (size_t i = 0; i <= kernel_length; i++)
        {
            // Read from global memory one by one. Shared memory can be used for optimization.
            if (idx_x + kernel_length < img_width && idx_y + kernel_length < img_length) {
                accumulator += abs(dev_img[(idx_y + j) * img_width + idx_x + i] - addernet_kernel[j * kernel_length + i]);
            }
        }
    }
    if (idx_x + kernel_radius < img_width && idx_y + kernel_radius < img_length) {
        dev_out[idx_x, idx_y] = accumulator;
    }
    __syncthreads();
}


int main()
{
    int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel if the image was RGB (not used)

    // Load a grayscale bmp image to an unsigned integer array with its height and weight.
    //  (uint8_t is an alias for "unsigned char")
    uint8_t* image = stbi_load("./samples/640x426.bmp", &width, &height, &bpp, NUM_CHANNELS);

    // Print for sanity check
    printf("Bytes per pixel: %d \n", bpp / 3); //Image is grayscale, so bpp / 3;
    printf("Height: %d \n", height);
    printf("Width: %d \n", width);
    printf("Number of threads: %d \n", NUM_THREADS);

    // Fill flattened kernel (random with seed)
    const int kernel_radius = KERNEL_RADIUS;
    const int kernel_size = pow(2 * kernel_radius + 1, 2);
    uint8_t* addernet_kernel = (uint8_t*)malloc(kernel_size);
    srand(1);
    for (size_t i = 0; i < kernel_size; i++)
    {
       addernet_kernel[i] = rand() % 50;
    }

    // Initialize 2D output array
    uint8_t* out_image = (uint8_t*)malloc(width * height);

    // Get timing info
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaError_t cudaStatus = addernetCUDA(out_image, image, addernet_kernel, kernel_radius, width, height);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution took %3.5f ms \n", time);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "contrastEnhancementCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // Write image array into a bmp file
    stbi_write_bmp("./out_img_640x426.bmp", width, height, 1, out_image);

    // Deallocate memory
    stbi_image_free(image);
    free(out_image);
    free(addernet_kernel);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addernetCUDA(uint8_t* out_img, uint8_t* img, uint8_t* addernet_kernel, const int kernel_radius, const int img_width, const int img_length)
{
    int blockSize = NUM_THREADS;
    int gridSize = img_width * img_length / blockSize + (img_width * img_length % blockSize != 0);

    // Temp CPU array that hold min values of each block. We need half of the gridSize since 
    uint8_t* min_array;
    min_array = (uint8_t*)malloc(ceil(gridSize / 2) * sizeof(uint8_t));
    // Device memory pointers for image and block minima
    uint8_t* dev_img = 0;
    uint8_t* dev_out = 0;
    uint8_t* dev_addernet_kernel = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    // Allocate GPU memory for the image and minima of seperate blocks
    cudaStatus = cudaMalloc((void**)&dev_img, img_width * img_length * sizeof(uint8_t));
    cudaStatus = cudaMalloc((void**)&dev_out, (img_width - 2 * kernel_radius) * (img_length - 2 * kernel_radius) * sizeof(uint8_t));
    cudaStatus = cudaMalloc((void**)&dev_addernet_kernel, pow(2 * kernel_radius + 1, 2) * sizeof(uint8_t));
    // Copy the image from host memory to GPU.
    cudaMemcpy(dev_img, img, img_width * img_length * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMempcy img failed!");
        goto Error;
    }


    dim3 grid, block;
    block.x = blockSize;
    grid.x = gridSize;

    addernetKernel << <grid, block >> > (dev_out, dev_img, dev_addernet_kernel, kernel_radius, img_width, img_length);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out_img, dev_out, (img_width - 2 * kernel_radius) * (img_length - 2 * kernel_radius) * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    return cudaStatus;

Error:
    cudaFree(dev_img);
    cudaFree(dev_out);
    cudaFree(dev_addernet_kernel);


    return cudaStatus;
}
