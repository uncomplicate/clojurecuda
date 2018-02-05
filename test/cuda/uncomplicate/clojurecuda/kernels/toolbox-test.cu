extern "C" {
        
    __global__ void sum_reduce (const int n, ACCUMULATOR* acc, const REAL* x) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        const ACCUMULATOR sum = block_reduction_sum( (gid < n) ? x[gid] : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }
    };

    __global__ void sum_reduce_horizontal (const int m, const int n, ACCUMULATOR* acc, const REAL* a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const int i = m * gid_1 + gid_0;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const ACCUMULATOR sum = block_reduction_sum_2( (valid) ? a[i] : 0.0);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = sum;
        }
    }

    __global__ void sum_reduce_vertical (const int m, const int n, ACCUMULATOR* acc, const REAL* a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const int i = n * gid_0 + gid_1;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const ACCUMULATOR sum = block_reduction_sum_2( (valid) ? a[i] : 0.0);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = sum;
        }
    }
}
