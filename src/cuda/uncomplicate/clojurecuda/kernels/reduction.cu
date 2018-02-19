extern "C" {

#ifndef REAL
#define REAL float
#endif
    
#ifndef ACCUMULATOR
#define ACCUMULATOR float
#endif
    
#ifndef WGS
#define WGS 1024
#endif

#ifndef WGSm
#define WGSm 64
#endif

#ifndef WGSn
#define WGSn 16
#endif

// ================= Sum reduction =============================================

    __device__ ACCUMULATOR block_reduction_sum (const ACCUMULATOR value) {

        const int local_id = threadIdx.x;

        __shared__ ACCUMULATOR lacc[WGS];
        lacc[local_id] = value;

        __syncthreads();

        ACCUMULATOR pacc = value;
        int i = blockDim.x;
        while (i > 0) {
            const bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
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

    __device__ ACCUMULATOR block_reduction_sum_2 (const ACCUMULATOR value) {

        const int local_row = threadIdx.x;
        const int local_col = threadIdx.y;
        const int local_m = blockDim.x;

        __shared__ ACCUMULATOR lacc[WGS];
        lacc[local_row + local_col * local_m] = value;

        __syncthreads();
        
        ACCUMULATOR pacc = value;
        int i = blockDim.y;
        while (i > 0) {
            const bool include_odd = (i > ((i >> 1) << 1)) && (local_col == ((i >> 1) - 1));
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

    __global__ void sum_reduction(const int n, ACCUMULATOR* acc) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        const ACCUMULATOR sum = block_reduction_sum( (gid < n) ? acc[gid] : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }
    }

    __global__ void sum_reduction_horizontal (const int m, const int n, ACCUMULATOR* acc) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const int i = m * gid_1 + gid_0;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const ACCUMULATOR sum = block_reduction_sum_2( (valid) ? acc[i] : 0.0);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = sum;
        }
    }

    __global__ void sum_reduction_vertical (const int m, const int n, ACCUMULATOR* acc) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const int i = n * gid_0 + gid_1;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const ACCUMULATOR sum = block_reduction_sum_2( (valid) ? acc[i] : 0.0);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = sum;
        }
    }

}
