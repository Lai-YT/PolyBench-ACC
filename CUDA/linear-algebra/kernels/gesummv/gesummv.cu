/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "gesummv.cuh"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

#define RUN_ON_CPU


void gesummv(int nn, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NN,NN,nn,nn), DATA_TYPE POLYBENCH_2D(B,NN,NN,nn,nn), DATA_TYPE POLYBENCH_1D(tmp,NN,nn),
		DATA_TYPE POLYBENCH_1D(x,NN,nn), DATA_TYPE POLYBENCH_1D(y,NN,nn))
{
	int i, j;
	
	for (i = 0; i < _PB_NN; i++)
	{
		tmp[i] = 0;
		y[i] = 0;
		for (j = 0; j < _PB_NN; j++)
		{
			tmp[i] = A[i][j] * x[j] + tmp[i];
			y[i] = B[i][j] * x[j] + y[i];
		}
		
		y[i] = alpha * tmp[i] + beta * y[i];
	}
}


void init(int nn, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A,NN,NN,nn,nn), DATA_TYPE POLYBENCH_2D(B,NN,NN,nn,nn),
	DATA_TYPE POLYBENCH_1D(x,NN,nn))
{
  	int i, j;

	*alpha = 43532;
	*beta = 12313;

 	for (i = 0; i < nn; i++)
    	{
    		x[i] = ((DATA_TYPE) i) / NN;
      	
		for (j = 0; j < nn; j++) 
		{
			A[i][j] = ((DATA_TYPE) i*j) / NN;
			B[i][j] = ((DATA_TYPE) i*j) / nn;
		}
    }
}


void compareResults(int nn, DATA_TYPE POLYBENCH_1D(y,NN,nn), DATA_TYPE POLYBENCH_1D(y_outputFromGpu,NN,nn))
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<nn; i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gesummv_kernel(int nn, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NN,NN,nn,nn), DATA_TYPE POLYBENCH_2D(B,NN,NN,nn,nn),
	DATA_TYPE POLYBENCH_1D(tmp,NN,nn), DATA_TYPE POLYBENCH_1D(x,NN,nn), DATA_TYPE POLYBENCH_1D(y,NN,nn))
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < _PB_NN)
	{
		int j;
		for(j = 0; j < _PB_NN; j++)
		{	
			tmp[i] += A[i][j] * x[j];
			y[i] += B[i][j] * x[j];
		}
		y[i] = alpha * tmp[i] + beta  * y[i];
	}
}

void gesummvCuda(int nn, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NN,NN,nn,nn), DATA_TYPE POLYBENCH_2D(B,NN,NN,nn,nn),
		DATA_TYPE POLYBENCH_1D(tmp,NN,nn), DATA_TYPE POLYBENCH_1D(x,NN,nn), DATA_TYPE POLYBENCH_1D(y,NN,nn),  
		DATA_TYPE POLYBENCH_1D(y_outputFromGpu,NN,nn))
{
	DATA_TYPE (*A_gpu)[NN];
	DATA_TYPE (*B_gpu)[NN];
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NN * NN);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NN * NN);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NN);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NN);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NN);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NN * NN, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NN * NN, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NN, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NN, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NN, cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)NN) / ((float)block.x) ), 1);


	/* Start timer. */
  	polybench_start_instruments;

	gesummv_kernel<<< grid, block>>>(nn, alpha, beta, A_gpu, B_gpu, tmp_gpu, x_gpu, y_gpu);
	cudaDeviceSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NN, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nn,
		 DATA_TYPE POLYBENCH_1D(y,NN,nn))

{
  int i;

  for (i = 0; i < nn; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int nn = NN;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NN,NN,nn,nn);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NN,NN,nn,nn);
	POLYBENCH_1D_ARRAY_DECL(tmp,DATA_TYPE,NN,nn);
	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,NN,nn);
	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,NN,nn);
	POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu,DATA_TYPE,NN,nn);

	init(nn, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x));
	
	GPU_argv_init();
	gesummvCuda(nn, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y),  
		POLYBENCH_ARRAY(y_outputFromGpu));
	
	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		gesummv(nn, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y));
		
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(nn, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(nn, POLYBENCH_ARRAY(y_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);  
	POLYBENCH_FREE_ARRAY(tmp);
	POLYBENCH_FREE_ARRAY(x);  
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(y_outputFromGpu);

	return 0;
}

#include <polybench.c>
