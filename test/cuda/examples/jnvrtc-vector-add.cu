/*
 * Created based on example from Marco Hutter:
 *
 * JCuda - Java bindings for NVIDIA CUDA
 *
 * Copyright 2008-2016 Marco Hutter - http://www.jcuda.org
 */

extern "C"
__global__ void add(int n, float *a, float *b, float *sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sum[i] = a[i] + b[i];
    }
};
