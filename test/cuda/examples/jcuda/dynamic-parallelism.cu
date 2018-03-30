/*
 * Created based on example from Marco Hutter:
 *
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */

extern "C"
__global__ void childKernel(unsigned int parentThreadIndex, float* data) {
    data[threadIdx.x] = parentThreadIndex + 0.1f * threadIdx.x;
}

extern "C"
__global__ void parentKernel(unsigned int size, float *data) {
    childKernel<<<1, 8>>>(threadIdx.x, data + threadIdx.x * 8);
}
