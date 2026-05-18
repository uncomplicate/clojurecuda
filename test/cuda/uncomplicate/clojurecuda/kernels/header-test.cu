#include <cuda_fp16.h>

extern "C" {

#include <stdint.h>

    __global__ void inc (int n, float* a) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            half temp = __float2half(a[i]);
            a[i] = 1.0 + __half2float(temp);
        }
    };

}
