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
  float *d_a, *d_b, *d_c;

  // Allocate host memory
  hipHostMalloc(&a, sizeof(float) * N);
  hipHostMalloc(&b, sizeof(float) * N);
  hipHostMalloc(&c, sizeof(float) * N);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // Allocate device memory
  hipMalloc((void **)&d_a, sizeof(float) * N);
  hipMalloc((void **)&d_b, sizeof(float) * N);
  hipMalloc((void **)&d_c, sizeof(float) * N);

  float gpu_elapsed_time_ms;

  // // Some events to count the execution time
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // Transfer data from host to device memory
  hipMemcpy(d_a, a, sizeof(float) * N, hipMemcpyHostToDevice);
  hipMemcpy(d_b, b, sizeof(float) * N, hipMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // Start to count execution time of GPU version
  hipEventRecord(start, 0);

  // Executing kernel
  vector_add<<<gridSize, blockSize>>>(d_c, d_a, d_b, N);

  // Time counting terminate
  hipEventRecord(stop, 0);
  hipDeviceSynchronize();
  hipEventSynchronize(stop);

  // Transfer data back to host memory
  hipMemcpy(c, d_c, sizeof(float) * N, hipMemcpyDeviceToHost);

  // Compute time elapse on GPU computing
  hipEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("Time elapsed on vector addition on GPU: %f ms.\n\n",
         gpu_elapsed_time_ms);

  // Copy data back to host memory
  hipMemcpy(c, d_c, sizeof(float) * N, hipMemcpyDeviceToHost);

  // Validation
  //   for (int i = 0; i < N; i++) {
  //     if (c[i] != a[i] + b[i]) {
  //       printf("Validation failed at element %d!\n", i);
  //       exit(-1);
  //     }
  //   }

  // Deallocate device memory
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);

  // Deallocate host memory
  hipHostFree(a);
  hipHostFree(b);
  hipHostFree(c);

  return 0;
}
