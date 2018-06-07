#include "cuda_common.h"
#include <cmath>
#include <string>

// Kernel function to add the elements of two arrays
//__global__
//void add(int n, float *x, float *y)
//{
//	for (int i = 0; i < n; i++)
//		y[i] = x[i] + y[i];
//}
//
//void addWithCuda(int N, float *x, float *y) {
//	add<<<1, 1>>>(N, x, y);
//}

//__global__
//void add(int n, float *x, float *y)
//{
//	int index = threadIdx.x;
//	int stride = blockDim.x;
//	for (int i = index; i < n; i += stride)
//		y[i] = x[i] + y[i];
//}
//
//void addWithCuda(int N, float *x, float *y) {
//	add<<<1, 256>>>(N, x, y);
//}

__global__
void add(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

__global__
void multiply(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] * y[i];
}

__global__
void substract(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] - y[i];
}

__global__
void divide(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] / y[i];
}

__global__
void sqrt(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = std::sqrtf(x[i]);
}

__global__
void power(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = std::powf(x[i], y[i]);
}

__global__
void sin(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = std::sinf(x[i]);
}

__global__
void cos(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = std::cosf(x[i]);
}

__global__
void tan(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		//y[i] = std::tanf(x[i]);
		y[i] = std::sinf(x[i]) / std::cosf(x[i]);
}

__global__
void ctg(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		//y[i] = 1.0f / std::tanf(x[i]);
		y[i] = std::cosf(x[i]) / std::sinf(x[i]);
}

__global__
void complex(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = std::sqrtf(std::cosf(x[i]) / std::sinf(x[i])) / std::sqrt(std::sinf(y[i]) / std::cosf(y[i]));
}

void doWithCuda(int N, float *x, float *y, OpType OT) {

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	switch (OT) {
	case ADD:
		add<<<numBlocks, blockSize>>>(N, x, y); break;
	case SUBSTRACT:
		substract<<<numBlocks, blockSize>>>(N, x, y); break;
	case MULTIPLY:
		multiply<<<numBlocks, blockSize>>>(N, x, y); break;
	case DIVIDE:
		divide<<<numBlocks, blockSize>>>(N, x, y); break;
	case SQRT:
		sqrt<<<numBlocks, blockSize>>>(N, x, y); break;
	//case POW:
	//	power<<<numBlocks, blockSize>>>(N, x, y); break;
	case SIN:
		sin<<<numBlocks, blockSize>>>(N, x, y); break;
	case COS:
		cos<<<numBlocks, blockSize>>>(N, x, y); break;
	case TAN:
		tan<<<numBlocks, blockSize>>>(N, x, y); break;
	case CTG:
		ctg<<<numBlocks, blockSize>>>(N, x, y); break;
	case COMPLEX:
		complex<<<numBlocks, blockSize>>>(N, x, y); break;
	}

	
}