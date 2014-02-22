/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#define TILE_WIDTH 16
#define BLOCK_WIDTH 16

///////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P, int M_row, int N_col, int width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	
	
	int tx = threadIdx.x; int ty = threadIdx.y; 
	int bx = blockIdx.x; int by = blockIdx.y;

	int col = bx * TILE_WIDTH + tx;
	int row = by * TILE_WIDTH + ty;
	float pvalue = 0;
	
	int num_tile =1 +((width-1) >> 4) ;  // Shift the Tile 
	for(int i=0; i < num_tile; i++)
	{
		if( TILE_WIDTH*i + tx < width && row < M_row )  // Make Sure Mds and Nds in the Valid Area

			Mds[ty][tx] = M.elements[ row * width  +tx+ i*TILE_WIDTH];
		else					
			Mds[ty][tx] = 0;  // divergence happens
	
		if( TILE_WIDTH * i + ty < width && col < N_col )
			Nds[ty][tx] = N.elements[(ty+i*TILE_WIDTH)* N_col +col];
		else
			Nds[ty][tx] = 0;

		__syncthreads();

		for(int k=0; k < TILE_WIDTH ; k++)
			pvalue += Mds[ty][k]*Nds[k][tx];
		__syncthreads();
	}
   	if(col<N_col && row<M_row)   // make sure [row,col] is in the valid area, divergence happens 
		P.elements[col+row*N_col] = pvalue;
	
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
