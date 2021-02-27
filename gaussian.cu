#include <stdio.h>

/* Function that normalizes the filter on the max element */
void gaussNorm(float *data,int size){
	float max=0;
	for (int i=0;i<size*size;i++)
		if(data[i]>max) max = data[i];
	for (int i=0;i<size*size;i++)
		data[i]/=max;
}

/* This funtion calculates the 2D gussian filter gaussFilter[size*size]
 * It is based on MATLAB's fspecial('gaussian',size,sigma) function's algorithm.
 */
float * gaussFilter(int size, float sigma){
	/* Get the mesh's rows and columns */
	float rows = ((float)size-1)/2;
	float cols = ((float)size-1)/2;
	/* Allocate the final array  */
	float *filter = (float *)malloc(size*size*sizeof(float));
	/* In the following loops a mesh is simulated from -rows to +rows and
      	 * from -cols to cols with step 1.
	 * Every element el[i][j]  of the 2D filter is calculated with the following formula
	 * el[i][j] = exp (- (x^2 + y^2) / (2*sigma^2)) / sum (el)
	 * where x and y are the mesh's elements for the specific filter's element.
	 */
	float sum = 0;
	float x=-rows;
	for (int i = 0; i<size; i++){
		float y=-cols;
		for (int j=0; j<size; j++){
			filter[i*size + j] = exp(- (x*x + y*y)/ (2*sigma*sigma));		
			sum += filter[i*size + j];
			/* Step = 1 */
			y++;
		}
		x++;
	}
	for (int i=0;i<size*size;i++)
		filter[i] =filter[i] / sum ;		

	return filter;
}
