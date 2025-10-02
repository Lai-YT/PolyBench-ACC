/*
 * Copyright (c) 2025 Lai-YT
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>

#include "gesummv.cuh"
#include "polybench.h"

namespace cublas {

#define CUBLAS_CHECK(status) \
	do { \
		cublasStatus_t stat = (status); \
		if (stat != CUBLAS_STATUS_SUCCESS) { \
			fprintf(stderr, "cuBLAS error: %d\n", stat); \
			exit(EXIT_FAILURE); \
		} \
	} while(0)

void gesummvCuda(int nn, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NN,NN,nn,nn), DATA_TYPE POLYBENCH_2D(B,NN,NN,nn,nn),
		DATA_TYPE POLYBENCH_1D(tmp,NN,nn), DATA_TYPE POLYBENCH_1D(x,NN,nn), DATA_TYPE POLYBENCH_1D(y,NN,nn),  
		DATA_TYPE POLYBENCH_1D(y_outputFromCublas,NN,nn))
{
	DATA_TYPE (*A_gpu)[NN];
	DATA_TYPE (*B_gpu)[NN];
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NN * NN);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NN * NN);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NN);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NN);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NN * NN, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NN * NN, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NN, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NN, cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)NN) / ((float)block.x) ), 1);

	cublasHandle_t handle;
	cublasCreate(&handle);

	/* Start timer. */
  	polybench_start_instruments;

	float zero = 0.0f;
	float one = 1.0f;

	/* y = beta * B * x + 0 * y */
	CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, NN, NN, &beta, reinterpret_cast<float*>(B_gpu), NN, reinterpret_cast<float*>(x_gpu), 1, &zero, reinterpret_cast<float*>(y_gpu), 1));
	/* y = alpha * A * x + y */
	CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, NN, NN, &alpha, reinterpret_cast<float*>(A_gpu), NN, reinterpret_cast<float*>(x_gpu), 1, &one, reinterpret_cast<float*>(y_gpu), 1));
	cudaDeviceSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cublasDestroy(handle);

	cudaMemcpy(y_outputFromCublas, y_gpu, sizeof(DATA_TYPE) * NN, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
}

#undef CUBLAS_CHECK

} // namespace cublas
