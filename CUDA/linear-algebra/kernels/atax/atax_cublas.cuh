/*
 * Copyright (c) 2025 Lai-YT
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cublas_v2.h>

#include "atax.cuh"
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

void ataxGpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny), DATA_TYPE POLYBENCH_1D(x, NX, nx), DATA_TYPE POLYBENCH_1D(y, NY, ny), 
		DATA_TYPE POLYBENCH_1D(tmp, NX, nx), DATA_TYPE POLYBENCH_1D(y_outputFromCublas, NY, ny))
{
	DATA_TYPE (*A_gpu)[NY];
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX);

	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);

	/* Start timer. */
  	polybench_start_instruments;

	// NOTE: cuBLAS kernels are dynamically loaded with `cudaLibraryLoadData` before the first call.
	// The measured time includes the overhead of loading the kernels, which can be much longer than the kernel execution time.

	float alpha = 1.0f;
	float beta = 0.0f;

	CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, NX, NY, &alpha, reinterpret_cast<float*>(A_gpu), NX, x_gpu, 1, &beta, tmp_gpu, 1));
	CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, NX, NY, &alpha, reinterpret_cast<float*>(A_gpu), NX, tmp_gpu, 1, &beta, y_gpu, 1));
	cudaDeviceSynchronize();

	/* Stop and print timer. */
	printf("cuBLAS Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cublasDestroy(handle);

	cudaMemcpy(y_outputFromCublas, y_gpu, sizeof(DATA_TYPE) * NY, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
}

#undef CUBLAS_CHECK

} // namespace cublas
