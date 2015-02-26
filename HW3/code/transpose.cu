#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_utils.h"
#include "timer.c"

typedef float dtype;
const int size_const = 32;
__global__ 
void matTrans(dtype* AT, dtype* A, int N)  {
	/* Fill your code here */

//	__shared__ float tileb[tile][tile];
   
   int tb_size = blockDim.x;
  int col = blockIdx.x * tb_size + threadIdx.x;
  int row = blockIdx.y * tb_size + threadIdx.y;
  int width = gridDim.x*tb_size; 
 int chunk = blockDim.y;

  __shared__ dtype ldata[size_const][size_const]; 

 //copy data from global to shared memory and transpose 
 for (int i = 0; i < tb_size; i += chunk){
      //if(col < N && row + i < N)
	ldata[threadIdx.y + i][threadIdx.x] = A[(row+i)*N + col];
	}
  __syncthreads();
  
   //since it's been transposed while copying, block ids are inverted
   col = blockIdx.y * tb_size + threadIdx.x;
   row = blockIdx.x * tb_size + threadIdx.y;
  
  //int indx1 = col + width * row;
  //int indx2 = row + width * col;
  
 for (int j = 0; j < tb_size; j += chunk)
       
	 AT[(row+j)*N + col] = ldata[threadIdx.x][threadIdx.y + j];
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
  dtype * d_A, *d_AT, *A2, *AT2;
 // int chunk,nThreads,tbSize,numTB;
	
  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();
        
	const int tb_x = 32; //size of each thread block
	const int tb_y = 8; //each thread blocks width
	

	dim3 numTB ;
	numTB.x = (int)ceil((double)N/(double)tb_x) ;
	numTB.y = (int)ceil((double)N/(double)tb_x) ;
	dim3 tbSize; //= (tile,rows);
	tbSize.x = tb_x;
	tbSize.y = tb_y;
	int d_N;
	if( N%32)d_N = numTB.x *tb_x;		
	else d_N =N;
        printf("d_n %d",d_N);
	stopwatch_start (timer);
	/* run your kernel here */
        CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_A, d_N * d_N  * sizeof (dtype)));
        A2 = (dtype *) malloc (d_N * d_N  * sizeof (dtype)); //padded A
	AT2 = (dtype*) malloc (d_N * d_N * sizeof(dtype)); //padded AT
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_AT, d_N * d_N * sizeof (dtype)));
	

	t_malloc = stopwatch_stop (timer);
	fprintf (stderr, "cudaMalloc: %Lg seconds\n", t_malloc);


	stopwatch_start (timer);
	// disperse the main array to a padded array
	
	for ( int count = 0; count < N ; count++)
	{
		for ( int j = 0; j < N ; j++)
		{
			A2[count*d_N+j] = A[count*N+j];
		} 
		for ( int k = 0; k < (d_N - N); k++)
		{
			A2[count* d_N + N + k] = 0;
		}  
	}
	for ( int count = N; count < d_N; count++)
	{
		for( int j =0; j < d_N; j++)
		{
			A2[count*d_N + j] = 0;
		}
	}
	
	// copy arrays to device via PCIe
	CUDA_CHECK_ERROR (cudaMemcpy (d_A, A2, d_N * d_N * sizeof (dtype), cudaMemcpyHostToDevice));
	
	//CUDA_CHECK_ERROR (cudaMemcpy (d_A, A, d_N * d_N * sizeof (dtype), cudaMemcpyHostToDevice));
	
	t_pcie = stopwatch_stop (timer);
	fprintf (stderr, "cudaMemcpy (and padding): %Lg seconds\n", t_pcie);


	stopwatch_start (timer);
	// kernel invocation
	matTrans<<<numTB,tbSize>>>(d_AT, d_A, d_N);
	cudaThreadSynchronize ();
	//if(! N%32){
     	CUDA_CHECK_ERROR (cudaMemcpy ( AT2,d_AT, d_N * d_N * sizeof (dtype),cudaMemcpyDeviceToHost));
	//combine the main array with removing/ignoring the empty boxes between each row. 
	for ( int count = 0; count < N ; count++)
	{
		for ( int j = 0; j < N ; j++)
		{
			AT[count*N+j] = AT2[count*d_N+j];
		} 
	}

	//else 
	//CUDA_CHECK_ERROR (cudaMemcpy (AT, d_AT, d_N * d_N * sizeof (dtype), cudaMemcpyDeviceToHost));
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
