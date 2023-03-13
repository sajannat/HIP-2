#include <hip/hip_runtime.h>
#include <time.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#define N 100000000

double GetTimeInSec() {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  double timeInSec = static_cast<double>(time.tv_sec) +
                     static_cast<double>(time.tv_nsec) * 1e-9;
  return timeInSec;
}

__global__ void vector_add(float *c, float *a, float *b, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

void cpu_vector_add(float *c, float *a, float *b, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;

  // Allocate host memory
  a = (float *)malloc(sizeof(float) * N);
  b = (float *)malloc(sizeof(float) * N);
  c = (float *)malloc(sizeof(float) * N);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // Allocate device memory
  HIP_ASSERT(hipMalloc((void **)&d_a, sizeof(float) * N));
  HIP_ASSERT(hipMalloc((void **)&d_b, sizeof(float) * N));
  HIP_ASSERT(hipMalloc((void **)&d_c, sizeof(float) * N));

  float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

  // Some events to count the execution time
  hipEvent_t start, stop;
  HIP_ASSERT(hipEventCreate(&start));
  HIP_ASSERT(hipEventCreate(&stop));

  // Transfer data from host to device memory
  HIP_ASSERT(hipMemcpy(d_a, a, sizeof(float) * N, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_b, b, sizeof(float) * N, hipMemcpyHostToDevice));

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  // Start to count execution time of GPU version
  hipEventRecord(start, 0);

  // Executing kernel
  vector_add<<<gridSize, blockSize>>>(d_c, d_a, d_b, N);

  // Time counting terminate
  HIP_ASSERT(hipEventRecord(stop, 0));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipEventSynchronize(stop));

  // Transfer data back to host memory
  HIP_ASSERT(hipMemcpy(c, d_c, sizeof(float) * N, hipMemcpyDeviceToHost));

  // Compute time elapse on GPU computing
  HIP_ASSERT(hipEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
  printf("Time elapsed on vector addition on GPU: %f ms.\n\n",
         gpu_elapsed_time_ms);

  // Start the CPU version
  double start_time = GetTimeInSec();

  cpu_vector_add(c, a, b, N);

  // Time counting terminate
  double end_time = GetTimeInSec();

  // compute time elapse on CPU computing
  printf("Time elapsed on vector addition on CPU: %f ms.\n\n",
         (end_time - start_time) * 1000);

  // Copy the data back from the GPU
  HIP_ASSERT(hipMemcpy(c, d_c, sizeof(float) * N, hipMemcpyDeviceToHost));

  // Validate the result
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
  free(a);
  free(b);
  free(c);

  return 0;
}
