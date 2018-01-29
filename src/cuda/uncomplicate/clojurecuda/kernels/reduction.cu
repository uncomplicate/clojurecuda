extern "C" {

#ifndef REAL
#define REAL float
#endif
    
#ifndef ACCUMULATOR
#define ACCUMULATOR REAL
#endif
    
#ifndef BLOCKS
#define BLOCKS 1024
#endif

#ifndef BLOCKSm
#define BLOCKSm 64
#endif

#ifndef BLOCKSn
#define BLOCKSn 16
#endif

// ================= Sum reduction =============================================

    __device__ ACCUMULATOR block_reduction_sum (const ACCUMULATOR value) {

        const int local_id = threadIdx.x;

        __shared__ ACCUMULATOR lacc[BLOCKS];
        lacc[local_id] = value;

        __syncthreads();

        ACCUMULATOR pacc = value;
        int i = blockDim.x;
        while (i > 0) {
            bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
            i >>= 1;
            if (include_odd) {
                pacc += lacc[local_id + i + 1];
            }
            if (local_id < i) {
                pacc += lacc[local_id + i];
                lacc[local_id] = pacc;
            }
            __syncthreads();
        }

        return lacc[0];
    }

    __device__ ACCUMULATOR block_reduction_sum_row (const ACCUMULATOR value) {

        const int local_row = threadIdx.x;
        const int local_col = threadIdx.y;
        const int local_m = blockDim.x;

        __shared__ ACCUMULATOR lacc[BLOCKSm * BLOCKSn];
        lacc[local_row + local_col * local_m] = value;

        __syncthreads();

        ACCUMULATOR pacc = value;
        int i = blockDim.y;
        while (i > 0) {
            bool include_odd = (i > ((i >> 1) << 1)) && (local_col == ((i >> 1) - 1));
            i >>= 1;
            if (include_odd) {
                pacc += lacc[local_row + (local_col + i + 1) * local_m];
            }
            if (local_col < i) {
                pacc += lacc[local_row + (local_col + i) * local_m];
                lacc[local_row + local_col * local_m] = pacc;
            }
            __syncthreads();
        }

        return lacc[local_row];

    }

    __device__ ACCUMULATOR block_reduction_sum_col (const ACCUMULATOR value) {

        const int local_row = threadIdx.y;
        const int local_col = threadIdx.x;
        const int local_m = blockDim.y;

        __shared__ ACCUMULATOR lacc[BLOCKSm * BLOCKSn];
        lacc[local_row + local_col * local_m] = value;

        __syncthreads();

        ACCUMULATOR pacc = value;
        int i = blockDim.x;
        while (i > 0) {
            bool include_odd = (i > ((i >> 1) << 1)) && (local_col == ((i >> 1) - 1));
            i >>= 1;
            if (include_odd) {
                pacc += lacc[local_row + (local_col + i + 1) * local_m];
            }
            if (local_col < i) {
                pacc += lacc[local_row + (local_col + i) * local_m];
                lacc[local_row + local_col * local_m] = pacc;
            }
            __syncthreads();
        }

        return lacc[local_row];

    }

    __global__ void sum_reduction(int n, ACCUMULATOR* acc) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        ACCUMULATOR sum = block_reduction_sum( (i < n) ? acc[i] : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }
    }

}
