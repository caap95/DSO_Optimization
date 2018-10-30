#ifndef CUDA_FUNCTIONS
#define CUDA_FUNCTIONS

Eigen::Vector3f * makeImagesGPU(int, int, int, int, int, Eigen::Vector3f*);

void calcResAndGSGPU();

#endif
