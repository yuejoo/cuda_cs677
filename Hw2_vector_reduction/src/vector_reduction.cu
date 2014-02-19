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

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <ctime>
// includes, project
//#include <cutil.h>

// includes, kernels
#include "vector_reduction_kernel.cu"
// For simplicity, just to get the idea in this MP, we're fixing the problem size to 512 elements.
//#define NUM_ELEMENTS 512

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name);
float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
	int 
main( int argc, char** argv) 
{
	runTest( argc, argv);
	return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
	void
runTest( int argc, char** argv) 
{
	//CUT_DEVICE_INIT();

	int num_elements = NUM_ELEMENTS;
	int errorM = 0;

	const unsigned int array_mem_size = sizeof( float) * num_elements;

	// allocate host memory to store the input data
	float* h_data = (float*) malloc( array_mem_size);

	// * No arguments: Randomly generate input data and compare against the 
	//   host's result.
	// * One argument: Read the input data array from the given file.
	switch(argc-1)
	{      
		case 1:  // One Argument
		errorM = ReadFile(h_data, argv[1]);
		if(errorM != 0)
		{
		   printf("Error reading input file!\n");
		   exit(1);
		}
		break;

		default:  // No Arguments or one argument
			// initialize the input data on the host to be integer values
			// between 0 and 1000
			time_t timer;
			srand((unsigned)time(&timer));

			for( unsigned int i = 0; i < num_elements; ++i) 
			{
				h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
			}
			break;  
	}
	// compute reference solution
	float reference = 0.0f;  
	computeGold(&reference , h_data, num_elements);
	// **===-------- Modify the body of this function -----------===**
	float result = computeOnDevice(h_data, num_elements);
	// **===-----------------------------------------------------------===**


	// We can use an epsilon of 0 since values are integral and in a range 
	// that can be exactly represented
	float epsilon = 0.0f;
	unsigned int result_regtest = (abs(result - reference) <= epsilon);
	printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
	printf( "device: %f  host: %f\n", result, reference);
	// cleanup memory
	free( h_data);
}

/*--make your own--
  int ReadFile(float* M, char* file_name)
  {
  unsigned int elements_read = NUM_ELEMENTS;
  if (cutReadFilef(file_name, &M, &elements_read, true))
  return 1;
  else
  return 0;
  }
 */
int ReadFile(float* M, char* file_name)
{
	FILE *f;
	unsigned int elements_read=NUM_ELEMENTS;
	if((f=fopen(file_name,"r"))==NULL)
		return 1;
	else
	{
		for(int i=0;i<elements_read;i++)
			fscanf(f,"%f",&M[i]);
		fclose(f);
		return 0;
	}
}


// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimentions, excutes kernel function, and copy result of scan back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice(float* h_data, int num_elements)
{

	// placeholder
	float *device_array = 0;
	float *host_array = h_data;
	float *device_out = 0;

        int num_output = num_elements/(BLOCK_SIZE<<1);
        if(num_elements % BLOCK_SIZE > 0)
                num_output++;

	float *host_out = (float*) malloc (num_output * sizeof(float));

	const unsigned int mem_array_size = sizeof(float)*num_elements;
	
	cudaMalloc((void**)&device_array, mem_array_size);
	cudaMalloc((void**)&device_out,sizeof(float)*num_output);

	if(host_array == 0 || device_array == 0 || device_out == 0)
	{
		printf("Can not allocate memory!");
		return 1.0f;
	}

	cudaMemcpy(device_array,host_array, mem_array_size, cudaMemcpyHostToDevice);

	dim3 dimGrid(num_output,1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);
	reduction<<<dimGrid,dimBlock>>>(device_array, device_out);

	cudaMemcpy(host_out, device_out, num_output*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(device_array);
	cudaFree(device_out);

	host_array[0]= 0;
	
	for(int i=0; i< num_output; i++)
		host_array[0]+=host_out[i];
	free(host_out);
	return host_array[0];

}
