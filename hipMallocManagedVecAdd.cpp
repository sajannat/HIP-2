#include <hip/hip_runtime.h>
#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#define N 100000000

__global__ void vector_add(float *c, float *a, float *b, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  float *a, *b, *c;

  // Allocate managed memory
  HIP_ASSERT(hipMallocManaged(&a, sizeof(float) * N));
  HIP_ASSERT(hipMallocManaged(&b, sizeof(float) * N));
  HIP_ASSERT(hipMallocManaged(&c, sizeof(float) * N));

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

  // Some events to count the execution time
  hipEvent_t start, stop;
  HIP_ASSERT(hipEventCreate(&start));
  HIP_ASSERT(hipEventCreate(&stop));

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // Start to count execution time of GPU version
  HIP_ASSERT(hipEventRecord(start, 0));

  // Executing kernel
  vector_add<<<gridSize, blockSize>>>(c, a, b, N);

  // Time counting terminate
  HIP_ASSERT(hipEventRecord(stop, 0));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipEventSynchronize(stop));

  // Compute time elapse on GPU computing
  HIP_ASSERT(hipEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
  printf("Time elapsed on vector addition on GPU: %f ms.\n\n",
         gpu_elapsed_time_ms);

  // Validation
  //   for (int i = 0; i < N; i++) {
  //     if (c[i] != a[i] + b[i]) {
  //       printf("Validation failed at element %d!\n", i);
  //       exit(-1);
  //     }
  //   }

  // Deallocate device memory
  hipFree(a);
  hipFree(b);
  hipFree(c);

  return 0;
}
