#include <stdlib.h>
#include <stdio.h>
#include "cuPrintf.cu"
#include "cuda_utils.h"
#include "timer.c"

typedef float dtype;


__global__ 
void matTrans(dtype* AT, dtype* A, int N,int chunk)  {
	/* Fill your code here */
	//int ThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	 int x = blockIdx.x * blockDim.x + threadIdx.x;
  	int y = blockIdx.y * blockDim.y + threadIdx.y;
 	 int width = N;
//  for (int j = 0; j < 32; j+= 8)
        AT[x*width + y] = A[y*width + x];
}
void
parseArg (int argc, char** argv, int* N)
{
	if(argc == 2) {
		*N = atoi (argv[1]);
		assert (*N > 0);
	} else {
		fprintf (stderr, "usage: %s <N>\n", argv[0]);
		exit (EXIT_FAILURE);
	}
}


void
initArr (dtype* in, int N)
{
	int i;

	for(i = 0; i < N; i++) {
		in[i] = (dtype) rand () / RAND_MAX;
	}
}

void
cpuTranspose (dtype* A, dtype* AT, int N)
{
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			AT[j * N + i] = A[i * N + j];
		}
	}
}

int
cmpArr (dtype* a, dtype* b, int N)
{
	int cnt, i;

	cnt = 0;
	for(i = 0; i < N; i++) {
		if(abs(a[i] - b[i]) > 1e-6) cnt++;
	}

	return cnt;
}



void
gpuTranspose (dtype* A, dtype* AT, int N)
{
  struct stopwatch_t* timer = NULL;
  long double t_gpu,t_malloc,t_pcie;
  dtype * d_A, *d_AT;
 // int chunk,nThreads,tbSize,numTB;
	
  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

  stopwatch_start (timer);
	/* run your kernel here */
        CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_A, N * N  * sizeof (dtype)));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_AT, N * N * sizeof (dtype)));
	t_malloc = stopwatch_stop (timer);
	fprintf (stderr, "cudaMalloc: %Lg seconds\n", t_malloc);


	stopwatch_start (timer);
	// copy arrays to device via PCIe
	CUDA_CHECK_ERROR (cudaMemcpy (d_A, A, N * N * sizeof (dtype), cudaMemcpyHostToDevice));
	t_pcie = stopwatch_stop (timer);
	fprintf (stderr, "cudaMemcpy: %Lg seconds\n", t_pcie);


	/* do not change this number */
//	nThreads = 1048576;
       
//	tbSize = 1024;
//	numTB = N;
  	int chunk ;//= N/tbSize;
	int tileSize = 32;
	dim3 numTB = (N/tileSize,N/tileSize,1);
	dim3 tbSize = (tileSize,tileSize,1);
	stopwatch_start (timer);
	// kernel invocation
	matTrans <<<numTB,tbSize>>> (d_AT, d_A, N,chunk);
	cudaThreadSynchronize ();
     	CUDA_CHECK_ERROR (cudaMemcpy ( AT,d_AT, N * N * sizeof (dtype),cudaMemcpyDeviceToHost));

//  cudaThreadSynchronize ();
  t_gpu = stopwatch_stop (timer);
  fprintf (stderr, "GPU transpose: %Lg secs ==> %Lg billion elements/second\n",
           t_gpu, (N * N) / t_gpu * 1e-9 );

}

int 
main(int argc, char** argv)
{
  /* variables */
	dtype *A, *ATgpu, *ATcpu;
  int err;

	int N;

  struct stopwatch_t* timer = NULL;
  long double t_cpu;


	N = -1;
	parseArg (argc, argv, &N);

  /* input and output matrices on host */
  /* output */
  ATcpu = (dtype*) malloc (N * N * sizeof (dtype));
  ATgpu = (dtype*) malloc (N * N * sizeof (dtype));

  /* input */
  A = (dtype*) malloc (N * N * sizeof (dtype));

	initArr (A, N * N);

	/* GPU transpose kernel */
	gpuTranspose (A, ATgpu, N);

  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

	stopwatch_start (timer);
  /* compute reference array */
	cpuTranspose (A, ATcpu, N);
  t_cpu = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute CPU transpose kernel: %Lg secs\n",
           t_cpu);

  /* check correctness */
	err = cmpArr (ATgpu, ATcpu, N * N);
	if(err) {
		fprintf (stderr, "Transpose failed: %d\n", err);
	} else {
		fprintf (stderr, "Transpose successful\n");
	}

	free (A);
	free (ATgpu);
	free (ATcpu);

  return 0;
}
