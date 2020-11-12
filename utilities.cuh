#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t arrayMalloc(void*** array, int length, size_t size);
cudaError_t arraycpyHtoD(void*** array_d, void*** array_h, int length, size_t size);
cudaError_t arraycpyDtoH(void*** array_h, void*** array_d, int length, size_t size);
cudaError_t oneMalloc(void** a_d, size_t size);
cudaError_t onecpyHtoD(void* dev_a, void* a, size_t size);
cudaError_t onecpyDtoH(void* a, void* dev_a, size_t size);
cudaError_t oneSetdevice();
cudaError_t oneLastError();
cudaError_t oneCudaDeviceSync();
