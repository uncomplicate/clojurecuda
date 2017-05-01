extern "C" {

    __global__ void sum_reduction(int n, float* acc) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            ACCUMULATOR sum = block_reduction_sum(acc[i]);
            if (threadIdx.x == 0) {
                acc[blockIdx.x] = sum;
            }
        }
    };

        
    __global__ void sum(int n, float* a, float* acc) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            ACCUMULATOR sum = block_reduction_sum(a[i]);
            if (threadIdx.x == 0) {
                acc[blockIdx.x] = sum;
            }
        }
    };

}
