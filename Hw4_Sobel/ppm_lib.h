
#ifndef max
#define max(a,b) ((a>b)?a:b)
#endif
#ifndef min
#define min(a,b) ((b>a)?a:b)
#endif
#define sqr(a) ((a)*(a))

unsigned char *read_pgm_image(char *fname, int *nx, int *ny);
int write_pgm_image(int *image, char *fname, int nx, int ny);
int write_pgm_Uimage(unsigned char *image, char *fname, int nx, int ny);

unsigned char *read_ppm_image(char *fname, int *nx, int *ny);
int write_ppm_Uimage(unsigned char *image, char *fname, int nx, int ny);
void ppm2rgb( unsigned char *image, unsigned char** red, unsigned char** green, unsigned char** blue,
			 int width, int height);
