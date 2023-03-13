#include <hip/hip_runtime.h>
#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define N 100000000

__global__ void vector_add(float *c, float *a, float *b, int n) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N){
        c[tid] = a[tid] + b[tid];
    }
}

int main(){
    float *a, *b, *c;

    // Allocate managed memory
    hipMallocManaged(&a, sizeof(float) * N);
    hipMallocManaged(&b, sizeof(float) * N);
    hipMallocManaged(&c, sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    // Some events to count the execution time
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Start to count execution time of GPU version
    hipEventRecord(start, 0);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Executing kernel 
    vector_add<<<gridSize, blockSize>>>(c, a, b, N);
    hipDeviceSynchronize();
    
    // Time counting terminate
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    // Compute time elapse on GPU computing
    hipEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on vector addition on GPU: %f ms.\n\n", gpu_elapsed_time_ms);
    
    printf("The first index of the resulting array, c[0] = %f\n", c[0]);

    // Deallocate device memory
    hipFree(a);
    hipFree(b);
    hipFree(c);

    return 0;
}
