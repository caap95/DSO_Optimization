#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <Eigen/Dense>

// CUDA runtime
#include </usr/local/cuda-9.0/include/cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include </usr/local/cuda-9.0/samples/common/inc/helper_functions.h>
#include </usr/local/cuda-9.0/samples/common/inc/helper_cuda.h>

#define K 16

__global__ void kernel_makeImagesGPU(int h, int w, int wlm1, int lvl, Eigen::Vector3f *input, Eigen::Vector3f *output)
{

    int i = blockIdx.x*K + threadIdx.x;
    int j = blockIdx.y*K + threadIdx.y;
    if(i < w && j < h)
    { 
        if(lvl == 2 && i == 0 && j == 0){
            float i1 = input[2*i   + 2*j*wlm1][0];
            float i2 = input[2*i+1 + 2*j*wlm1][0];
            float i3 = input[2*i   + 2*j*wlm1+wlm1][0];
            float i4 = input[2*i+1 + 2*j*wlm1+wlm1][0];
            float out = 0.25f*(i1 + i2 + i3 + i4);
            printf("h:%i w:%i wlm1:%i i1:%f i2:%f i3:%f i4:%f out:%f \n", h, w, wlm1, i1, i2, i3, i4, out);
        }
        output[i + j*w][0] = 0.25f * (input[2*i   + 2*j*wlm1][0] +
                                        input[2*i+1 + 2*j*wlm1][0] +
                                        input[2*i   + 2*j*wlm1+wlm1][0] +
                                        input[2*i+1 + 2*j*wlm1+wlm1][0]);    

    }
    
}
 

Eigen::Vector3f * makeImagesGPU(int h, int w, int hlm1, int wlm1, int lvl, Eigen::Vector3f* r)
{

    //printf("CUDA makeImages\n");
    int N = ( (h > w) ? h : w);
    dim3 blocks(N/K, N/K);
    dim3 threads(K, K);

    Eigen::Vector3f *dinput = NULL;
    Eigen::Vector3f *doutput = NULL;

    Eigen::Vector3f *res = new Eigen::Vector3f[w*h];

    /*int dout = sizeof(Eigen::Vector3f) * hlm1 * wlm1;
    printf("size:%i", dout);

    int x;
    std::cin >> x;*/

    checkCudaErrors(cudaMalloc((void **)&dinput, sizeof(Eigen::Vector3f) * hlm1 * wlm1));
    checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(Eigen::Vector3f) * h * w));
    checkCudaErrors(cudaMemcpy(dinput, r, sizeof(Eigen::Vector3f) * hlm1 * wlm1, cudaMemcpyHostToDevice));

    kernel_makeImagesGPU<<<blocks, threads>>>(h, w, wlm1, lvl, dinput, doutput);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(res, doutput, sizeof(Eigen::Vector3f) * h * w, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dinput));
    checkCudaErrors(cudaFree(doutput));

    return res;

}

void calcResAndGSGPU()
{

    

}