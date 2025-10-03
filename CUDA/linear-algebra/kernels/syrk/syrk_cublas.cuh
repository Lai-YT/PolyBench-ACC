/*
 * Copyright (c) 2025 Lai-YT
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cublas_v2.h>

#include "polybench.h"
#include "syrk.cuh"

namespace cublas {

#define CUBLAS_CHECK(status) \
	do { \
		cublasStatus_t stat = (status); \
		if (stat != CUBLAS_STATUS_SUCCESS) { \
			fprintf(stderr, "cuBLAS error: %s\n", cublasGetStatusString(stat)); \
			exit(EXIT_FAILURE); \
		} \
	} while(0)

void syrkCuda(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), 
		DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni)) {
	DATA_TYPE(* A_gpu)[NJ];
	DATA_TYPE(* C_gpu)[NI];

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NI);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NI, cudaMemcpyHostToDevice);
	
	cublasHandle_t handle;
	cublasCreate(&handle);

	/* Start timer. */
  	polybench_start_instruments;

	/* Since C is always taken as column-major in cuBLAS, we compute
	 * C^T = alpha * A^T * A + beta * C^T instead. */
	CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, NI, NJ, &alpha, reinterpret_cast<float *>(A_gpu), NJ, &beta, reinterpret_cast<float *>(C_gpu), NI));
	cudaDeviceSynchronize();

	/* Stop and print timer. */
	printf("cuBLAS Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cublasDestroy(handle);

	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NI, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(C_gpu);
}

#undef CUBLAS_CHECK

} // namespace cublas
