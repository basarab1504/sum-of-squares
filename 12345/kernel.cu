#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void squareKernel(int* data, int N);

int main(int argc, char** argv)
{
	int* h_data;
	int* d_data;
	//���������� ���������
	int n = 1000;
	//����� ���������
	int sum = 0;

	// �������� page-locked ������ �� �����
	// ��� ������� ����� ����� ������������ �������� ��� ��������� ������������� �������� ��� ������ ������� ����� ������ � �����������.
	cudaHostAlloc(&h_data, n * sizeof(int), cudaHostAllocPortable);

	// �������� ������ �� ����������
	cudaMalloc(&d_data, n * sizeof(int));
	
	dim3 block(512);
	dim3 grid((n + block.x - 1) / block.x);

	//grid - ���������� ������
	//block - ������ �����
	squareKernel<<<grid, block>>>(d_data, n);

	//�������� ������ � ���������� (d_data) �� ���� (h_data)
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
	//threadIdx � ����� ���� � �����
	//blockIdx � ����� �����, � ������� ��������� ����
	//blockDim � ������ �����

	//���������� ������ ���� ������ ����
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		data[i] = i * i;
	}
}