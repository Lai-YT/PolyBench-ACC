/**
 * mvt.cuh: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#ifndef MVT_CUH
# define MVT_CUH

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(N)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#define N 1024
#  endif

#  ifdef SMALL_DATASET
#define N 2048
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#define N 4096
#  endif

#  ifdef LARGE_DATASET
#define N 8192
#  endif

#  ifdef EXTRALARGE_DATASET
#define N 16384
#  endif
# endif /* !N */

# define _PB_N POLYBENCH_LOOP_BOUND(N,n)

# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

/* Thread block dimensions */
#ifndef DIM_THREAD_BLOCK_X
/* Each SM has 4 SMSPs, each of them can handle a warp of 32 threads.
 * So, to fully utilize the GPU, we need to have at least 128 threads
 * per block.  */
#define DIM_THREAD_BLOCK_X 128
#endif
#ifndef DIM_THREAD_BLOCK_Y
/* NOTE: The kernel doesn't use the Y dimension;
 * not setting it to 1 leads to redundant work being done. */
#define DIM_THREAD_BLOCK_Y 1
#endif

#endif /* !MVT*/
