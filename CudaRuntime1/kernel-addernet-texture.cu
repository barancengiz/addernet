#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define NUM_THREADS 1024
#define NUM_CHANNELS 1
// Random sample from the image for debug purposes.
#define DEBUG_IMG_IDX 20
// AdderNET kernel size
#define KERNEL_RADIUS 5
#define KERNEL_SIZE (2*KERNEL_RADIUS+1)*(2*KERNEL_RADIUS+1)   // number of elements in the kernel

typedef unsigned char uint8_t;

#include "stb_image.h"
#include "stb_image_write.h"
#include <cstdint>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define GPUTHREADSIZE 1024

__constant__ uint8_t addernet_const_kernel[KERNEL_SIZE];

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
//    else{
//        fprintf(stderr, "GPUassertNOERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
//    }
}

cudaError_t
addernetCUDA(uint8_t *out_img, uint8_t *img, uint8_t *addernet_kernel, int kernel_radius, int img_width,
             int img_height);

// Assuming VALID padding (size decreases)
// Index: kernel_radius -> img_width - kernel_radius - 1
// using square kernels like 5x5 (radius 2), 11x11 (radius 5)
// Single input & output channels
__global__ void
addernetKernel(uint8_t *dev_out, cudaTextureObject_t dev_img, const int kernel_radius,
               const int img_width, const int imgHeigth) {

    unsigned int tid_x = threadIdx.x;
    unsigned int tid_y = threadIdx.y;
    // Kernel radius can be added in the loop instead of using minus values.
    unsigned int idx_x = blockIdx.x * (blockDim.x) + threadIdx.x;
    auto row = idx_x / img_width;
    auto col = idx_x % img_width;
//    unsigned int idx_y = blockIdx.y * (blockDim.y * 2) + threadIdx.y;

    int accumulator = 0;
    int kernel_length = 2 * kernel_radius + 1;

    for (size_t j = 0; j < kernel_length; j++) {
        for (size_t i = 0; i < kernel_length; i++) {
            // Read from global memory one by one. Shared memory can be used for optimization.
            if (col + kernel_length <= img_width && row + kernel_length <= imgHeigth) {
//                int i1 = -abs(tex2D<uint8_t>(dev_img, col + i, row + j) - addernet_const_kernel[j * kernel_length + i]);
//                uint8_t i2 = tex2D<uint8_t>(dev_img, col + i, row + j) * addernet_const_kernel[j * kernel_length + i]
//                accumulator += -abs(tex2D<uint8_t>(dev_img, col + i, row + j) - addernet_const_kernel[j * kernel_length + i]);
                accumulator += tex2D<uint8_t>(dev_img, col + i, row + j) * addernet_const_kernel[j * kernel_length + i];
            }
        }
    }
    if (col + kernel_length <= img_width && row + kernel_length <= imgHeigth) {
        dev_out[(row * (img_width - kernel_radius * 2)) + col] = accumulator;
    }
    __syncthreads();
}

// Matrix summation where each thread covers an element each
__global__ void mat_mean(const long *mean_vals, long *mean_outs, int tot_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tot_size) return;

    //Copy memory
    extern __shared__ long means[];
    means[threadIdx.x] = mean_vals[idx];

    __syncthreads();

    //Reduce max

    int tid = threadIdx.x;
    if (blockDim.x > 1) {
        unsigned int lastDim = blockDim.x;
        for (unsigned int i = (blockDim.x + 1) / 2; i >= 1; (++i) >>= 1) {
            if (i == 1) {
                means[0] += means[1];
                __syncthreads();
                break;
            } else if (((lastDim & 0x1) == 1 && tid < (i - 1)) || (tid < i)) {
                means[tid] += means[tid + i];
            }
//            if (vec_idx < i-1 && i!=1) {
//                means[vec_idx] += means[vec_idx + i];
//            } else if (i == 1) {
//                means[0] += means[1];
//                __syncthreads();
//                break;
//            }
            (++lastDim) >>= 1;
            __syncthreads();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        mean_outs[blockIdx.x] = means[0];
    }

}

__global__ void square_sum(const uint8_t *mean_vals, long *mean_outs, long mean, int tot_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tot_size) return;
    long diff = mean_vals[idx] - mean;
    mean_outs[idx] = diff * diff;
}

__global__ void normalize(const uint8_t *img, uint8_t *out_img, long mean, long var, int tot_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tot_size) return;
    long diff = img[idx] - mean;
    out_img[idx] = diff / sqrtf(float(var));
}


int main() {
    int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel if the image was RGB (not used)
    float total_time = 0;
    // Load a grayscale bmp image to an unsigned integer array with its height and weight.
    //  (uint8_t is an alias for "unsigned char")
    int num_of_runs = 10;
    for (int run = 0; run < num_of_runs; run++) {
        uint8_t *image = stbi_load("../CudaRuntime1/samples/5184x3456.bmp", &width, &height, &bpp, NUM_CHANNELS);

        // Print for sanity check
        printf("Bytes per pixel: %d \n", bpp / 3); //Image is grayscale, so bpp / 3;
        printf("Height: %d \n", height);
        printf("Width: %d \n", width);
        printf("Number of threads: %d \n", NUM_THREADS);

        // Fill flattened kernel (random with seed)
        const int kernel_radius = KERNEL_RADIUS;
        const int kernel_size = pow(2 * kernel_radius + 1, 2);
        auto *addernet_kernel = (uint8_t *) malloc(kernel_size);
        srand(1);
        for (size_t i = 0; i < kernel_size; i++) {
            if (i == kernel_size / 2) {
                addernet_kernel[i] = 1;
            } else {
                addernet_kernel[i] = 0;
            }
        }
//        addernet_kernel[0]=0;
//        addernet_kernel[1]=0.2;
//        addernet_kernel[2]=0.14;
//        addernet_kernel[3]=0.14;

        // Initialize 2D output array
        uint8_t *out_image = (uint8_t *) malloc((width - 2 * kernel_radius) * (height - 2 * kernel_radius));

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
        total_time += time;
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
        stbi_write_bmp("./out_img_5184x3456.bmp", width - 2 * kernel_radius, height - 2 * kernel_radius, 1, out_image);

        // Deallocate memory
        stbi_image_free(image);
        free(out_image);
        free(addernet_kernel);
    }
    printf("Execution took %3.5f ms \n", total_time);
    printf("AVG: %3.5f ms \n", total_time / num_of_runs);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t
addernetCUDA(uint8_t *out_img, uint8_t *img, uint8_t *addernet_kernel, const int kernel_radius, const int img_width,
             const int img_length) {
    int blockSize = NUM_THREADS;
    int gridSize = img_width * img_length / blockSize + (img_width * img_length % blockSize != 0);

    // Device memory pointers for image
    uint8_t *dev_out;
    uint8_t *dev_addernet_kernel;
    cudaError_t cudaStatus;
    cudaArray_t cuArray;

    // Choose which GPU to run on, change this on a multi-GPU system.
//    gpuErrchk(cudaSetDevice(0));

    // CUDA Channel Descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

    // Allocate CUDA array in device memory
    gpuErrchk(cudaMallocArray(&cuArray, &channelDesc, img_width, img_length));

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed to by src
    const size_t spitch = img_width * sizeof(uint8_t);

    // Specify texture
    cudaResourceDesc resDesc{};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    cudaTextureDesc texDesc{};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    gpuErrchk(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // Copy data from host memory to CUDA array.
    gpuErrchk(cudaMemcpy2DToArray(cuArray, 0, 0, img, spitch, img_width * sizeof(uint8_t), img_length,
                                  cudaMemcpyHostToDevice));

    // Allocate GPU memory for the output
    int width = img_width - 2 * kernel_radius;
    int height = img_length - 2 * kernel_radius;
    gpuErrchk(cudaMalloc((void **) &dev_out,
                         width * height * sizeof(uint8_t)))

    // Initialize constant memory in GPU
    gpuErrchk(cudaMemcpyToSymbol(addernet_const_kernel, addernet_kernel, sizeof(uint8_t) * KERNEL_SIZE));

    dim3 grid, block;
    block.x = blockSize;
    grid.x = gridSize;

    addernetKernel <<<grid, block >>>(dev_out, texObj, kernel_radius, img_width, img_length);
    gpuErrchk(cudaGetLastError())

//    int n_blocks = 0;
//    long *mean_h, *var_h;
//    long *mean_d, *var_d;
//
//
//    size_t tot_size_1 = width * height * sizeof(long);
//    cudaMalloc((void **) &mean_d, tot_size_1);
//    cudaMalloc((void **) &var_d, tot_size_1);
//    cudaMemset(mean_d, 0, tot_size_1);
//    cudaMemset(var_d, 0, tot_size_1);
//    for (int i = 0; i < width * height; i++) {
//        cudaMemcpy(mean_d + i, dev_out, 1, cudaMemcpyDeviceToDevice);
//        cudaMemcpy(var_d + i, dev_out, 1, cudaMemcpyDeviceToDevice);
//    }
//    mean_h = (long *) malloc(sizeof(long));
//    var_h = (long *) malloc(sizeof(long));
//    int N = height * width;
//    do {
//        long *mean_temp_d;
//
//        int block_size = min(GPUTHREADSIZE, N);
//        n_blocks = N / block_size + (N % block_size == 0 ? 0 : 1);
//        size_t count = n_blocks * sizeof(long);
//        gpuErrchk(cudaMalloc((void **) &mean_temp_d, count));
////        gpuErrchk(cudaMalloc((void **) &min_temp_d, count));
//        gpuErrchk(cudaMemset(mean_temp_d, 0, count));
////        cudaMemset(min_temp_d, 255, count);
//        size_t i = block_size * sizeof(long);
//        mat_mean<<< n_blocks, block_size, i>>>(mean_d, mean_temp_d, N);
//        gpuErrchk(cudaGetLastError())
////        mat_min <<< n_blocks, block_size, block_size >>>(min_d, min_temp_d, N);
////        err = cudaGetLastError();
////        gpuErrchk(err)
//        cudaDeviceSynchronize();
////        gpuErrchk(cudaMemcpy(min_d, min_temp_d, count, cudaMemcpyDeviceToDevice));
//        gpuErrchk(cudaMemcpy(mean_d, mean_temp_d, count, cudaMemcpyDeviceToDevice));
//        cudaDeviceSynchronize();
//
////        gpuErrchk(cudaFree(min_temp_d));
//        gpuErrchk(cudaFree(mean_temp_d));
//        N = n_blocks;
//    } while (n_blocks != 1);
//    cudaMemcpy(mean_h, mean_d, sizeof(long), cudaMemcpyDeviceToHost);
////    cudaMemcpy(var_h, min_d, sizeof(long), cudaMemcpyDeviceToHost);
//    N = width * height;
//    *mean_h = *mean_h / N;
//    printf("mean: %d\n", *mean_h);
//    int block_size = min(GPUTHREADSIZE, N);
//    n_blocks = N / block_size + (N % block_size == 0 ? 0 : 1);
//    square_sum<<< n_blocks, block_size>>>(dev_out, var_d, *mean_h, N);
//
//    do {
//        long *mean_temp_d;
//
//        block_size = min(GPUTHREADSIZE, N);
//        n_blocks = N / block_size + (N % block_size == 0 ? 0 : 1);
//        size_t count = n_blocks * sizeof(long);
//        gpuErrchk(cudaMalloc((void **) &mean_temp_d, count));
////        gpuErrchk(cudaMalloc((void **) &min_temp_d, count));
//        gpuErrchk(cudaMemset(mean_temp_d, 0, count));
////        cudaMemset(min_temp_d, 255, count);
//        size_t i = block_size * sizeof(long);
//        mat_mean<<< n_blocks, block_size, i>>>(var_d, mean_temp_d, N);
//        gpuErrchk(cudaGetLastError())
////        mat_min <<< n_blocks, block_size, block_size >>>(min_d, min_temp_d, N);
////        err = cudaGetLastError();
////        gpuErrchk(err)
//        cudaDeviceSynchronize();
////        gpuErrchk(cudaMemcpy(min_d, min_temp_d, count, cudaMemcpyDeviceToDevice));
//        gpuErrchk(cudaMemcpy(var_d, mean_temp_d, count, cudaMemcpyDeviceToDevice));
//        cudaDeviceSynchronize();
//
////        gpuErrchk(cudaFree(min_temp_d));
//        gpuErrchk(cudaFree(mean_temp_d));
//        N = n_blocks;
//    } while (n_blocks != 1);
//    cudaMemcpy(var_h, var_d, sizeof(long), cudaMemcpyDeviceToHost);
////    cudaMemcpy(var_h, min_d, sizeof(long), cudaMemcpyDeviceToHost);
//    N = width * height;
//    printf("var: %d\n", *var_h);
//    *var_h = *var_h / N;
//    printf("var: %d\n", *var_h);
//
//    N = width * height;
//    block_size = min(GPUTHREADSIZE, N);
//    n_blocks = N / block_size + (N % block_size == 0 ? 0 : 1);
//    normalize<<< n_blocks, block_size>>>(dev_out, dev_out, *mean_h, *var_h, N);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out_img, dev_out,
                            width * height * sizeof(uint8_t),
                            cudaMemcpyDeviceToHost);

    // Destroy texture object
    gpuErrchk(cudaDestroyTextureObject(texObj));
    gpuErrchk(cudaFree(dev_out));

    return cudaStatus;
}
