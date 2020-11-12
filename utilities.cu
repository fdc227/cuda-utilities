#include "utilities.cuh"


cudaError_t arrayMalloc(void*** array, int length, size_t size)
{
	cudaError_t cudaStatus;

	for (int i = 0; i < length; i++)
	{
		cudaStatus = cudaMalloc(array[i], size);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!\n");
			goto Error;
		}
	}
Error:
	for (size_t i = 0; i < length; i++)
	{
		cudaFree(array[i]);
	}

	return cudaStatus;
}

cudaError_t arraycpyHtoD(void*** array_d, void*** array_h, int length, size_t size)
{
	cudaError_t cudaStatus;

	for (int i = 0; i < length; i++)
	{
		cudaStatus = cudaMemcpy(*array_d[i], *array_h[i], size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy HostToDevice of array %d failed!\n", i);
			fprintf(stderr, "Reasons for failure : %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
	}
Error:
	for (size_t i = 0; i < length; i++)
	{
		cudaFree(array_d[i]);
	}

	return cudaStatus;
}

cudaError_t arraycpyDtoH(void*** array_h, void*** array_d, int length, size_t size)
{
	cudaError_t cudaStatus;

	for (int i = 0; i < length; i++)
	{
		cudaStatus = cudaMemcpy(*array_h[i], *array_d[i], size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy DeviceToHost of array %d failed!\n", i);
			goto Error;
		}
	}
Error:
	for (size_t i = 0; i < length; i++)
	{
		cudaFree(array_d[i]);
	}

	return cudaStatus;
}

cudaError_t oneMalloc(void** a_d, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&a_d, size );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
	}
	return cudaStatus;
}

cudaError_t onecpyHtoD(void* dev_a, void* a, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}
	return cudaStatus;
}

cudaError_t onecpyDtoH(void* a, void* dev_a, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
	}
	return cudaStatus;
}

cudaError_t oneSetdevice()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
	}
	return cudaStatus;
}

cudaError_t oneLastError()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	return cudaStatus;
}

// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.

cudaError_t oneCudaDeviceSync()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	return cudaStatus;
}