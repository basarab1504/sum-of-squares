#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void squareKernel(int* data, int N);

int main(int argc, char** argv)
{
	int* h_data;
	int* d_data;
	//количество квадратов
	int n = 1000;
	//сумма квадратов
	int sum = 0;

	// выдел€ем page-locked пам€ть на хосте
	// эту функцию лучше всего использовать экономно дл€ выделени€ промежуточных областей дл€ обмена данными между хостом и устройством.
	cudaHostAlloc(&h_data, n * sizeof(int), cudaHostAllocPortable);

	// выдел€ем пам€ть на устройстве
	cudaMalloc(&d_data, n * sizeof(int));
	
	dim3 block(512);
	dim3 grid((n + block.x - 1) / block.x);

	//grid - количество блоков
	//block - размер блока
	squareKernel<<<grid, block>>>(d_data, n);

	//копируем данные с устройства (d_data) на хост (h_data)
	cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

	for (int j = 0; j < n; j++)
	{
		sum = sum + h_data[j];
	}

	printf("sum = %d\n", sum);
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