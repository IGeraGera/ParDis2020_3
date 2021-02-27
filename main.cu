#include <stdio.h>
#include <sys/time.h>
#include "inc/gaussian.h"
#include "inc/iofun.h"
#include "inc/imgfun.h"

/* CUDA Kernels */
/* Multiply each row of the unrolled data with the gauss filter using grid-stride*/
__global__
void gaussianPass(int patchSize, int dataSize, float *gaussFilter,float *data){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < dataSize; i += stride){
	     data[i] = gaussFilter[i%(patchSize*patchSize)] * data[i];
	}
}
/* Fill the matrix with the distances */
__global__
void distanceMatCalc(int totalPixels, int patchSize, float *distMat, float *data,float filtSig){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i = index; i < totalPixels*totalPixels; i += stride){
	     	int data_i = i / totalPixels;
		int data_j = i % totalPixels;
		float tmp = 0.0;
		/* if not ont diagonal */
	       	if (data_i != data_j){
			for(int elem = 0 ; elem <patchSize*patchSize ; elem++){
				float diff = (data[data_i*patchSize*patchSize + elem] - data[data_j*patchSize*patchSize + elem]);
				tmp += diff * diff;
			}
			tmp = exp(-tmp/(filtSig));
		//	tmp = exp(-tmp/(filtSig*filtSig));
		}	
		distMat[i]=tmp;
	}
}	
/* Find sum of rows for the distance matrix and divide each row element with it 
 * Put the max element of each row to the diagonal */
__global__
void distanceMatFinal(int totalPixels, float *distMat){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < totalPixels; i += stride){
		float sum = 0.0;
		float max = 0.0;
		for (int j = 0; j < totalPixels; j++){
			float element = distMat[i*totalPixels+j];
			/* Check if data is max */
			if(element>max) max = element;
			/* Add to sum to divide with it */
			sum += element;
		}
		sum += max;
		/* Iterate row again, put max to diagonal and divide with sum */
		for (int j = 0; j < totalPixels; j++){
			if (i == j ) distMat[i*totalPixels+j] = max/sum;
			else distMat[i*totalPixels+j] /= sum;
		}
	}
}
/* Vector Matrix Multiplication */
__global__
void vectorMatrixMult(int totalPixels, float* matrix,float *vector, float *out){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < totalPixels; i += stride){
		float sum =0.0;
		for (int j = 0; j < totalPixels; j++){
			sum += matrix[i*totalPixels+j]*vector[j];
		}
		out[i]=sum;
	}
	
}
/* for each distance matrix row find max and divide  */
int
main(int argc, char *argv[]){
	/* Start Timing*/
	struct timeval startTime,endTime;
	gettimeofday(&startTime,NULL);
	float patchSig = 5.0/3.0;
	float filtSig = 0.02;
	/* Check arguments if correct*/
	checkArgsNum(2,argc);
	/* Declare Image variables */	
 	int rawImgSize_i,rawImgSize_j;
	float *rawimage;
	/* Get image from argument given */
	rawimage = getImg(argv[1],&rawImgSize_i, &rawImgSize_j);
	normalizeImage(rawimage,rawImgSize_i,rawImgSize_j);
	/* Patch Size */	
	int patchSize = atoi(argv[2]);
	checkPatchSize(patchSize);
	/* Pad Image */
	int padImgSize_i,padImgSize_j;
	float *paddedImg;
	paddedImg = padImg(rawimage,rawImgSize_i,rawImgSize_j,patchSize,&padImgSize_i,&padImgSize_j);
	/* Get patches from image to rows */
	float *patchImg;
	patchImg = unrollPatches(paddedImg,rawImgSize_i,rawImgSize_j,patchSize);
	/* Calculate the gaussian Filter */
	float *gFilter = gaussFilter(patchSize,patchSig);
	//gaussNorm(gFilter,patchSize);
	
	/* Multiply gaussian filter with data */
	/* Memory for Kernel */
	float *kernel_gFilter, *kernel_data;
	int totalData = rawImgSize_i*rawImgSize_j*patchSize*patchSize;
		/*CUDA code */
		cudaMalloc(&kernel_gFilter,patchSize*patchSize*sizeof(float));
		cudaMalloc(&kernel_data, totalData*sizeof(float));
		cudaMemcpy(kernel_gFilter,gFilter,patchSize*patchSize*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(kernel_data,patchImg,totalData*sizeof(float),cudaMemcpyHostToDevice);	
		gaussianPass<<<(totalData+255)/256,256>>>(patchSize,totalData,kernel_gFilter,kernel_data);
	/* Find Distances matrix */
	/* Allocate distance matrix */
	float *kernel_distMat;
	int totalPixels =  rawImgSize_i*rawImgSize_j;
		/*CUDA code */
		cudaMalloc(&kernel_distMat,totalPixels*totalPixels*sizeof(float));
		distanceMatCalc<<<(totalPixels*totalPixels+255)/256,256>>>(totalPixels,patchSize,kernel_distMat,kernel_data,filtSig);
	/* Find sum of rows for the distance matrix and divide each row element with it 
	 * Put the max element of each row to the diagonal */
		distanceMatFinal<<<(totalPixels+255)/256,256>>>(totalPixels,kernel_distMat);
	/*Allocate memory for filtered output */
	float *filteredimage = (float *)malloc(totalPixels*sizeof(float));
	float *kernel_filteredimage;
	float *kernel_rawimage;
		cudaMalloc( &kernel_rawimage,totalPixels*sizeof(float));
		cudaMalloc( &kernel_filteredimage,totalPixels*sizeof(float));
		cudaMemcpy(kernel_rawimage,rawimage,totalPixels*sizeof(float),cudaMemcpyHostToDevice);

		vectorMatrixMult<<<(totalPixels+255)/256,256>>>(totalPixels,kernel_distMat,kernel_rawimage,kernel_filteredimage);

		cudaMemcpy(filteredimage,kernel_filteredimage,totalPixels*sizeof(float),cudaMemcpyDeviceToHost);

		/* Cuda Freeing memory */
		cudaFree(kernel_gFilter);
		cudaFree(kernel_data);
		cudaFree(kernel_distMat);
		cudaFree(kernel_rawimage);
		cudaFree(kernel_filteredimage);
	/* End Timer*/
	gettimeofday(&endTime,NULL);

	/* Write image */
	writeImg(argv[1],filteredimage,rawImgSize_i,rawImgSize_j);

	/* Free */ 
	free(patchImg);
	free(rawimage);
	free(filteredimage);
	free(gFilter);
	free(paddedImg);

	double timeInterval = (endTime.tv_sec * 1000000 + endTime.tv_usec) -
		(startTime.tv_sec * 1000000 + startTime.tv_usec);
	printf("Execution Time : %.2f us",timeInterval);

}
