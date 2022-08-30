#include<stdio.h>
# include "nn_cuda.h"

__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
        // printf("%d\n",c[i]);
    }
}
__global__ void helloFromGPU(void)
{
  printf("Hello World from GPU！\n");
}

__global__ void VecAdd(int* A, int* B, int* C)
{

    int i = threadIdx.x;
    C[i] = A[i] + B[i];

    
}

void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    helloFromGPU<<<1,10>>>();
    using namespace std;
    cout<<sizeof(c)<<endl;

    add2_kernel<<<grid, block>>>(c, a, b, n);
    

    // const int N=5;
    float A[n]={0};
    float B[n]={0};
    float C[n]={0};

    float *dev_a = 0;
    float *dev_b = 0;
    // float *dev_c = 0;

    // cudaSetDevice(0);
    // cudaMalloc((void**)&dev_c, n * sizeof(float));
    // cudaMalloc((void**)&dev_a, n * sizeof(float));
    // cudaMalloc((void**)&dev_b, n * sizeof(float));
    // cudaMemcpy(dev_a, A, n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_b, B, n * sizeof(float), cudaMemcpyHostToDevice);

    // VecAdd<<<1, n>>>(dev_a, dev_b, dev_c);

    // cudaGetLastError();
    // cudaDeviceSynchronize();
    cudaMemcpy(C, c, n * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaFree(dev_c);
    // cudaFree(dev_a);
    // cudaFree(dev_b);

    for (int i = 0; i < n; i++)
    {
        if (i!=0) printf(" ");
        printf("%d",C[i]);
        if (i==n-1)printf("\n");
    }
    
}