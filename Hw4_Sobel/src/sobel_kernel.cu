#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sobel_kernel.h"
#define BLOCK_WIDTH 16

__device__ int compute_sobel(int m00, int m01, int m02,
		int m10,          int m12,
		int m20, int m21, int m22,
		int thresh
		)
{
	int Horz = -m02 - 2*m12 - m22 + m00 + 2 * m10 + m20;
	int ver = m22 - m00 - 2*m01 - m02 + m20 + 2 * m21;
	int sum = ver * ver + Horz * Horz;
	if( sum > thresh )
		return 0xff;
	else
		return 0;

}


__global__ void sobel(int *input, int *output, int w, int h, int thresh)
{
	int idx = threadIdx.x+1;
	int idy = threadIdx.y+1;
	int px = threadIdx.x+blockDim.x*blockIdx.x;
	int py = threadIdx.y+blockDim.y*blockIdx.y;
	int bx = blockIdx.x*BLOCK_WIDTH;
	int by = blockIdx.y*BLOCK_WIDTH;

	unsigned int pd = px + py * w;

	__shared__ int share_map[BLOCK_WIDTH+2][BLOCK_WIDTH+2];


	if(threadIdx.y < 2)
	{
		share_map[threadIdx.x][threadIdx.y*(BLOCK_WIDTH+1)] = input[(by-1+ threadIdx.y*(1+BLOCK_WIDTH) )*w + px-1];
		share_map[threadIdx.x+BLOCK_WIDTH+1][threadIdx.y*(BLOCK_WIDTH+1)] = input[(by-1+ threadIdx.y*(1+BLOCK_WIDTH) )*w + px + BLOCK_WIDTH];
		share_map[idx][threadIdx.y*(BLOCK_WIDTH+1)] = input[(by-1+threadIdx.y*(1+BLOCK_WIDTH))*w+px];
		share_map[threadIdx.y*(BLOCK_WIDTH+1)][idx] = input[(by+threadIdx.x)*w+bx-1+threadIdx.y*(BLOCK_WIDTH+1)];

	}

		share_map[threadIdx.x+1][threadIdx.y+1] = input[pd];

	__syncthreads();
	if(px<w && py<h)
	{
		if(px>0 && px < w-1 && py>0 && py<h-1 )
			output[pd] =	
				compute_sobel(
						share_map[idx-1][idy-1],
						share_map[idx][idy-1],
						share_map[idx+1][idy-1],
						share_map[idx-1][idy],
						share_map[idx+1][idy],
						share_map[idx-1][idy+1],
						share_map[idx][idy+1],
						share_map[idx+1][idy+1],
						thresh
					     );
		else
			output[pd]=0;	
	}
}

extern "C" __host__ int* callkernel(unsigned int* input, int width,int height,int thresh)
{

	int *d_input = 0;
	int *d_output = 0;
	int num_input = width*height;
	int *h_output = (int *)malloc(num_input*sizeof(int));
	const unsigned int mem_size = num_input*sizeof(int);
	cudaMalloc((void**)&d_input,mem_size);
	cudaMalloc((void**)&d_output,mem_size);
	printf("Gridsize:%d %d\n",(width-1)/BLOCK_WIDTH+1,(height-1)/BLOCK_WIDTH+1);
	cudaMemcpy(d_input,input, mem_size,cudaMemcpyHostToDevice);

	dim3 gridsize((width-1)/BLOCK_WIDTH+1,(height-1)/BLOCK_WIDTH+1);
	dim3 blocksize(BLOCK_WIDTH,BLOCK_WIDTH);
	sobel<<< gridsize,blocksize >>>(d_input,d_output,width,height,thresh);

	cudaMemcpy(h_output,d_output, mem_size,cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
	printf("GPU Done!\n");	
	return h_output;
}
