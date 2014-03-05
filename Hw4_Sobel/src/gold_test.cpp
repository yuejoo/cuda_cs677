#include<stdio.h>
#include<stdlib.h>


int* gold_test(unsigned int *pic, int *src,  const int& xsize, const int& ysize, const int& maxval, const int& thresh, int& flag)
{
	int numbytes =  xsize * ysize * 3 * sizeof( int );
	int *result = (int *) malloc( numbytes );
	if (!result) { 
		fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
		exit(-1); // fail
	}

	int i, j, magnitude, sum1, sum2; 
	int *out = result;

	for (int col=0; col<ysize; col++) {
		for (int row=0; row<xsize; row++) { 
			*out++ = 0; 
		}
	}

	for (i = 1;  i < ysize - 1; i++) {
		for (j = 1; j < xsize -1; j++) {

			int offset = i*xsize + j;

			sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ] 
				+ 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
				+     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];

			sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
				- pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];

			magnitude =  sum1*sum1 + sum2*sum2;

			if (magnitude > thresh)
				result[offset] = 255;
			else 
				result[offset] = 0;
			if(src[offset]!=result[offset])
				flag = 0;
		}
	}

	fprintf(stderr, "CPU Done!\n"); 
	return result;

}
