/**
 *  \file mandelbrot--mpi.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

// root sends linse to each processor (SCATTER)
// processor calculates mandelbrot
// processor sends result to root
// root renders image (GATHER)
 
/**
 *  *  \file mandelbrot--mpi.cc
 *   *
 *    *  \brief Implement your parallel mandelbrot set in this file.
 *     */

#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include "render.hh"
#include "timer.c"
using namespace std;

#define WIDTH 1000
#define HEIGHT 1000

int
mandelbrot(double x, double y) {
  int maxit = 511;
  double cx = x;
  double cy = y;
  double newx, newy;

  int it = 0;
  for (it = 0; it < maxit && (x*x + y*y) < 4; ++it) {
    newx = x*x - y*y + cx;
    newy = 2*x*y + cy;
    x = newx;
    y = newy;
  }
  return it;
}

int
main(int argc, char* argv[]) {
  double minX = -2.1;
  double maxX = 0.7;
  double minY = -1.25;
  double maxY = 1.25;
  
  int root = 0;

  int height, width;
  if (argc == 3) {
    height = atoi (argv[1]);
    width = atoi (argv[2]);
    assert (height > 0 && width > 0);
  } else {
    fprintf (stderr, "usage: %s <height> <width>\n", argv[0]);
    fprintf (stderr, "where <height> and <width> are the dimensions of the image.\n");
    return -1;
  }

  double it = (maxY - minY)/height;
  double jt = (maxX - minX)/width;
  double x, y; 
   MPI_Init(NULL,NULL);
  int count,col,row,rank,size;
  double time = 0.0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  int* global_m; //Where results are gathered
   int N = floor(height/size); /*number of rows assigned to each processor*/
  
  if(rank == 0)
	global_m = (int*) malloc(sizeof(int)*width*height);


   MPI_Barrier(MPI_COMM_WORLD);
   time -= MPI_Wtime();
  int* local_m = (int*) malloc(sizeof(int)*width*N); //each threads calculation result
/* calculate mandelbrot for all the rows assigned, store in a 1D array*/
  for(count = 0; count < width * N; count++)
  {
  	col = count % width;
  //	row = floor(count/width) + rank * N; //Intern Joe
  	int row = (rank + (int)floor(count/width)*size); //Intern Susie
  	local_m[count] = mandelbrot(minX + col * jt , minY + row * it);
     
  }
 
//printf("Proc %d got here with : %d",rank,local_m[0]);
  MPI_Barrier(MPI_COMM_WORLD);
//int stride_l,stride_g;

  MPI_Gather(local_m , N*width, MPI_INT, global_m, N*width, MPI_INT, 0, MPI_COMM_WORLD);
//free(local_m);
 MPI_Barrier(MPI_COMM_WORLD);
 time += MPI_Wtime();
  gil::rgb8_image_t img(height, width);
  auto img_view = gil::view(img);

if(rank == 0){
printf("parallel time: %f",time);
//for(int c =0; c< width*height; c++)
//cout << global_m[c];	
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      img_view(i,j) = render(global_m[width*((int)floor(j/size)+N*(j%size)) + i]/512.0); //Intern Susie
    //  img_view(i,j) = render(global_m[width*j + i]/512.0); //Intern Joe
    }
  }
free(global_m);
}
  gil::png_write_view("mandelbrot.png", const_view(img));
  MPI_Finalize();
//long double t =stopwatch_stop(timer);
//printf("Parallel time : %Lg",t);

}
/* eof */

