#include <stdlib.h>
#include <stdio.h>

#include "cuda_utils.h"
#include "timer.c"

typedef float dtype;

__global__ 
void matTrans(dtype* AT, dtype* A, int N)  {
	/* Fill your code here */

//	__shared__ float tileb[tile][tile];
   
  int tb_size = blockDim.x;
  int col = blockIdx.x * tb_size + threadIdx.x;
  int row = blockIdx.y * tb_size + threadIdx.y;
  int width = gridDim.x*tb_size; 
  //for (int j = 0; j < tile; j += rows)
    // tileb[threadIdx.y+j][threadIdx.x] = A[(row+j)*width + col];

  int chunk = blockDim.y;
  int indx1 = col + width * row;
  int indx2 = row + width * col;
  
 for (int j = 0; j < tb_size; j += chunk)
       
	 AT[indx2 + j] = A[indx1 + j * width];
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
        
	const int tb_x = 32; //size of each thread block
	const int tb_y = 8; //each thread blocks width
	dim3 numTB ;
	numTB.x = ceil(N/tb_x);
	numTB.y = ceil(N/tb_x);
	dim3 tbSize; //= (tile,rows);
	tbSize.x = tb_x;
	tbSize.y = tb_y;
	
	
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


	stopwatch_start (timer);
	// kernel invocation
	matTrans<<<numTB,tbSize>>>(d_AT, d_A, N);
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
