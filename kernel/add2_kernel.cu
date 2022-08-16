#include<stdio.h>

__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}
__global__ void helloFromGPU(void)
{
  printf("Hello World from GPU！\n");
}

__global__ void VecAdd(int* A, int* B, int* C)
{
    for(int j=0 ;j<100000000;j++){
        int i = threadIdx.x;
        C[i] = A[i] + B[i];
    }
    
}

void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    helloFromGPU<<<1,10>>>();

    // add2_kernel<<<grid, block>>>(c, a, b, n);
    const int N=5;
    int A[N]={1,2,3,4,5};
    int B[N]={2,2,2,2,2};
    int C[N]={0};

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    cudaSetDevice(0);
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMemcpy(dev_a, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, B, N * sizeof(int), cudaMemcpyHostToDevice);

    VecAdd<<<1, N>>>(dev_a, dev_b, dev_c);

    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaMemcpy(C, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    for (int i = 0; i < N; i++)
    {
        if (i!=0) printf(" ");
        printf("%d",C[i]);
        if (i==N-1)printf("\n");
    }
    
}