#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>

__global__ void squareKernel(int* data, int N);

int main(int argc, char** argv)
{
	int* arr;
	int* h_data;
	int* d_data;
	//количество квадратов + 1
	int n = 256;
	//сумма квадратов
	int sum = 0;
	int sumCPU = 0;

	//эвенты дл€ подсчета времени выполнени€ на GPU
	cudaEvent_t g_start, g_stop;
	cudaEventCreate(&g_start);
	cudaEventCreate(&g_stop);
	//clock дл€ подсчета времени выполнени€ на CPU
	clock_t c_start, c_stop;

	// выдел€ем page-locked пам€ть на хосте
	// эту функцию лучше всего использовать экономно дл€ выделени€ промежуточных областей дл€ обмена данными между хостом и устройством.
	cudaHostAlloc(&h_data, n * sizeof(int), cudaHostAllocPortable);

	//cudaMemcpy(h_data, arr, n * sizeof(int), cudaMemcpyHostToDevice);

	// выдел€ем пам€ть на устройстве
	cudaMalloc(&d_data, n * sizeof(int));

	dim3 block(512);
	dim3 grid((n + block.x - 1) / block.x);

	//grid - количество блоков
	//block - размер блока
	squareKernel <<<grid, block>>>(d_data, n);

	cudaEventRecord(g_start, 0);

	//копируем данные с устройства (d_data) на хост (h_data)
	cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(g_stop, 0);
	cudaEventSynchronize(g_stop);
	float GPUelapsedTime;
	cudaEventElapsedTime(&GPUelapsedTime, g_start, g_stop);

	for (int j = 0; j < n; j++)
	{
		sum = sum + h_data[j];
	}

	//CPU
	double CPUelapsedTime;
	c_start = clock();
	for (int i = 0; i < n; i++)
	{
		sumCPU += i * i;
	}
	c_stop = clock();
	CPUelapsedTime = ((double)(c_stop - c_start)) / (CLOCKS_PER_SEC * 1000);

	printf("Time in GPU: %lfms\n", GPUelapsedTime);
	printf("Time in CPU: %lfms\n", CPUelapsedTime);
	printf("sum GPU for %d = %d\n", n-1, sum);
	printf("sum CPU for %d = %d\n", n-1, sumCPU);

	cudaEventDestroy(g_start);
	cudaEventDestroy(g_stop);
	return 0;
}

__global__ void squareKernel(int* data, int N)
{
	//threadIdx Ц номер нити в блоке
	//blockIdx Ц номер блока, в котором находитс€ нить
	//blockDim Ц размер блока

	//глобальный индекс нити внутри сети
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		data[i] = i * i;
	}
}