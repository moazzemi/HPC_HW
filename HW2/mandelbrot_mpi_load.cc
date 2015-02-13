/**
 *  \file mandelbrot--mpi.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

// root sends linse to each processor
// Each processor computes and sends result to root
// if not finished root sends it another line
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
#define DIE 0
#define DATA 1


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


void root(int w, int h, int size){
int lines_rem = size -1;
int rank,row;
int * line = (int*) malloc(sizeof(int)*(w+1));
//suppose each processor first computes line = its rank-1
int next_line = size-1;
MPI_Status stat;
int * global = (int*) malloc(sizeof(int)*w*h);
int proc_killed =0;
 gil::rgb8_image_t img(h, w);
  auto img_view = gil::view(img);


while(lines_rem > 0 )
{
	//do this while there are lines left to compute
	MPI_Recv(line, w+1,MPI_INT,MPI_ANY_SOURCE, DATA ,MPI_COMM_WORLD,&stat);//recieve from any processor, then get its ID
	//find out the rank
	rank = stat.MPI_SOURCE;
	lines_rem--;
	//the line number
	row = line[w];
	//printf("root recieved line %d eq to %d from %d size is : %d ",row,line[0],rank,size);
	//memcpy(global+row,line,sizeof(int)*w);
	
	for( int i = row; i < row + w; i++){
	//	cout << global[i];
		//global[i] = line [i - row]; 
		//change the previous line to comment to next line for renderin
		img_view(i-row,row) = render(line[i-row]/512.0);
	//	cout << global[i];
	}
//	cout<<endl;
	//store line in global
	if(next_line < h){
	//send next_line to rank
	MPI_Send(&next_line,1,MPI_INT,rank,DATA,MPI_COMM_WORLD);//just send them the line number
	next_line++;
	lines_rem++;
	}
	else{
	//send terminate signal to rank
	MPI_Send(0,0,MPI_INT,rank,DIE,MPI_COMM_WORLD);
	}

}

/*
for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
    //  img_view(i,j) = render(global_m[width*((int)floor(j/size)+N*(j%size)) + i]/512.0); //Intern Susie
//         cout << global[w*j + i];
	img_view(i,j) = render(global[w*j + i]/512.0);
  	//cout<<global[w*j+i];
	  }
  }
*/
 gil::png_write_view("mandelbrot.png", const_view(img));

return ;
}
void proc(int width,int height,int rank){
	 double minX = -2.1;
  	double maxX = 0.7;
  	double minY = -1.25;
  	double maxY = 1.25;
         double it = (maxY - minY)/height;
         double jt = (maxX - minX)/width;
	  

	int* line = (int *) malloc(sizeof(int)*(width+1));
	//compute line = rank -1;
	int row = rank -1;
	int count;
	MPI_Status stat;
      for(count = 0; count < width ; count++)
 	 {
  	line[count] = mandelbrot(minX + count * jt , minY + row * it);
  	}
	line[width]=row;
  
	//send it to root
	//cout << row << "*";
	MPI_Send(line,width+1,MPI_INT,0,DATA,MPI_COMM_WORLD);
	//do this while signal terminate recieved from root:
	while(1){
	   //recieve line number from root
	   MPI_Recv(&row,1,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,&stat);
	   if(stat.MPI_TAG == DIE || row >= height)
		{//printf ("proc %d killed",rank);
		return;}
	   //send result back to root
	   else{
		//printf("Processor %d got line %d",rank,row);
		for(count = 0; count<width; count++)
		{
			line[count] = mandelbrot(minX + count * jt , minY + row *it);
			//cout<< line[count];
		}
		line[width] = row;
		MPI_Send(line,width+1,MPI_INT,0,DATA,MPI_COMM_WORLD);
	   }
	}
	return;
}



int
main(int argc, char* argv[]) {
/*  double minX = -2.1;
  double maxX = 0.7;
  double minY = -1.25;
  double maxY = 1.25;
  
  int root = 0;
*/
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

 /* double it = (maxY - minY)/height;
  double jt = (maxX - minX)/width;
  double x, y; */
   MPI_Init(NULL,NULL);
  int rank,size;
  double time = 0.0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  int* global_m; //Where results are gathered
 //  int N = floor(height/size); /*number of rows assigned to each processor*/
  
 

   MPI_Barrier(MPI_COMM_WORLD);
   time -= MPI_Wtime();
 
 
//printf("Proc %d got here with : %d",rank,local_m[0]);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0){
	// global_m = (int*) malloc(sizeof(int)*width*height);
	root(width,height,size);
	}
   else{
	//printf("proc %d got here",rank);
	proc(width,height,rank);
	}
  
  MPI_Barrier(MPI_COMM_WORLD);
  time += MPI_Wtime();
 
if(rank == 0){
printf("parallel time with dynamic load: %f",time);
//for(int c =0; c< width*height; c++)
//cout << global_m[c];	
 // free(global_m);

 }
  MPI_Finalize();
//long double t =stopwatch_stop(timer);
//printf("Parallel time : %Lg",t);

}
/* eof */

