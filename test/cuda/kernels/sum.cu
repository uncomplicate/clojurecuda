extern "C" {
        
    __global__ void sum(const int n, const REAL* a, ACCUMULATOR* acc) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        ACCUMULATOR sum = block_reduction_sum( (i < n) ? a[i] : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }
    };

}
