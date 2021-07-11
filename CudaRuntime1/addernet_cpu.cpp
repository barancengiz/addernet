//
// Created by kadircanbecek on 3.07.2021.
//
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define NUM_THREADS 1024
#define NUM_CHANNELS 1
// Random sample from the image for debug purposes.
#define DEBUG_IMG_IDX 20
// AdderNET kernel size
#define KERNEL_RADIUS 5
#define KERNEL_SIZE (2*KERNEL_RADIUS+1)*(2*KERNEL_RADIUS+1)

typedef unsigned char uint8_t;

#include "stb_image.h"
#include "stb_image_write.h"
#include <cstdint>
#include <chrono>
#include <iostream>


void conv2d(uint8_t *out_img, const uint8_t *img, const uint8_t *addernet_kernel, const int kernel_radius,
            const int img_width,
            const int img_height) {
    auto kernel_width = kernel_radius * 2 + 1;
    int new_h = img_height - kernel_radius * 2;
    int new_w = img_width - kernel_radius * 2;
    for (int i = 0; i < new_h; i++) {
        for (int j = 0; j < new_w; j++) {
            uint8_t accumulator = 0;
            for (int k = 0; k < kernel_width; k++) {
                for (int m = 0; m < kernel_width; m++) {
//                    uint8_t i1 = img[(i + k) * img_width + j + m] * addernet_kernel[k * kernel_width + m];
//                    uint8_t i2 = -abs(img[(i + k) * img_width + j + m] - addernet_kernel[k * kernel_width + m]);
                    accumulator += img[(i + k) * img_width + j + m] * addernet_kernel[k * kernel_width + m];
//                    accumulator += -abs(img[(i + k) * img_width + j + m] - addernet_kernel[k * kernel_width + m]);
                }
            }
            out_img[i * new_w + j] = accumulator;
        }
    }
}

int main() {
    int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel if the image was RGB (not used)
    float total_time = 0.0f;  //bytes per pixel if the image was RGB (not used)

    int num_of_runs = 10;
    for (int run = 0; run < num_of_runs; run++) {
        uint8_t *image = stbi_load("../CudaRuntime1/samples/5184x3456.bmp", &width, &height, &bpp, NUM_CHANNELS);

        // Print for sanity check
        printf("Bytes per pixel: %d \n", bpp / 3); //Image is grayscale, so bpp / 3;
        printf("Height: %d \n", height);
        printf("Width: %d \n", width);

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
        auto start = std::chrono::high_resolution_clock::now();
        conv2d(out_image, image, addernet_kernel, kernel_radius, width, height);
        auto stop = std::chrono::high_resolution_clock::now();

        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        float msecs = microseconds.count() / 1000.0f;
        std::cout << "Time: " << msecs << " microseconds\n";
        total_time += msecs;
        printf("\n");

        stbi_write_bmp("./out_img_5184x3456_cpu.bmp", width - 2 * kernel_radius, height - 2 * kernel_radius, 1,
                       out_image);

        // Deallocate memory
        stbi_image_free(image);
        free(out_image);
        free(addernet_kernel);
    }
    printf("Total Time: %f milliseconds\n", total_time);
    printf("Avg Time: %f milliseconds\n", total_time / num_of_runs);
}

