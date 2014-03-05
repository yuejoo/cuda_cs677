

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
#include "gold_test.cpp"

#define DEFAULT_THRESHOLD  20000

#define DEFAULT_FILENAME "1.ppm"

extern "C" int* callkernel(unsigned int*,int,int,int);
unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval )
{

	if ( !filename || filename[0] == '\0') {
		fprintf(stderr, "read_ppm but no file name\n");
		return NULL;  // fail
	}

	FILE *fp;

	fprintf(stderr, "read_ppm( %s )\n", filename);
	fp = fopen( filename, "rb");
	if (!fp) 
	{
		fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
		return NULL; // fail 

	}

	char chars[1024];
	//int num = read(fd, chars, 1000);
	int num = fread(chars, sizeof(char), 1000, fp);

	if (chars[0] != 'P' || chars[1] != '6') 
	{
		fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
		return NULL;
	}

	unsigned int width, height, maxvalue;


	char *ptr = chars+3; // P 6 newline
	if (*ptr == '#') // comment line! 
	{
		ptr = 1 + strstr(ptr, "\n");
	}

	num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
	fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
	*xsize = width;
	*ysize = height;
	*maxval = maxvalue;

	unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
	if (!pic) {
		fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
		return NULL; // fail but return
	}

	// allocate buffer to read the rest of the file into
	int bufsize =  3 * width * height * sizeof(unsigned char);
	if ((*maxval) > 255) bufsize *= 2;
	unsigned char *buf = (unsigned char *)malloc( bufsize );
	if (!buf) {
		fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
		return NULL; // fail but return
	}





	// TODO really read
	char duh[80];
	char *line = chars;

	// find the start of the pixel data.   no doubt stupid
	sprintf(duh, "%d\0", *xsize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *ysize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *maxval);
	line = strstr(line, duh);


	fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
	line += strlen(duh) + 1;

	long offset = line - chars;
	//lseek(fd, offset, SEEK_SET); // move to the correct offset
	fseek(fp, offset, SEEK_SET); // move to the correct offset
	//long numread = read(fd, buf, bufsize);
	long numread = fread(buf, sizeof(char), bufsize, fp);
	fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

	fclose(fp);


	int pixels = (*xsize) * (*ysize);
	for (int i=0; i<pixels; i++) pic[i] = (int) buf[3*i];  // red channel



	return pic; // success
}

void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
	FILE *fp;
	int x,y;

	fp = fopen(filename, "wb");
	if (!fp) 
	{
		fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
		exit(-1); 
	}



	fprintf(fp, "P6\n"); 
	fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);

	int numpix = xsize * ysize;
	for (int i=0; i<numpix; i++) {
		unsigned char uc = (unsigned char) pic[i];
		fprintf(fp, "%c%c%c", uc, uc, uc); 
	}
	fclose(fp);

}




int main( int argc, char **argv )
{

	int thresh = DEFAULT_THRESHOLD;
	char *filename;
	filename = strdup( DEFAULT_FILENAME);

	if (argc > 1) {
		if (argc == 3)  { // filename AND threshold
			filename = strdup( argv[1]);
			thresh = atoi( argv[2] );
		}
		if (argc == 2) { // default file but specified threshhold

			thresh = atoi( argv[1] );
		}

		fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
	}


	int xsize, ysize, maxval;
	int flag=1;
	unsigned int *pic = read_ppm( filename, &xsize, &ysize, &maxval ); 
	int *res = callkernel(pic,xsize,ysize,thresh);
	int *result = gold_test(pic,res,xsize,ysize,maxval, thresh,flag);
//	int x= BLOCK_WIDTH;
	if(flag)
		printf("TEST PASS!\n");
	else
		printf("TEST FAIL!\n");
	write_ppm( "cpu_result.ppm", xsize, ysize, 255, result);
	write_ppm( "gpu_result.ppm", xsize, ysize, 255, res);
	free(res);
	free(result);
}

