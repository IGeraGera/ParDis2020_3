#include <stdio.h>
/* Function that normalizes the image */
void
normalizeImage(float *image,int Size_i,int Size_j){
	int Size = Size_i*Size_j;
	/* Find MIN */
	float min = FLT_MAX;
	for(int i=0;i<Size;i++){
		if (image[i]<min) min = image[i];
	}
	/* Subtract min */
	for(int i=0;i<Size;i++)
		image[i] -= min;
	/* Find Max*/
	float max =0.0;
	for(int i=0;i<Size;i++){
		if (image[i]>max) max = image[i];
	}
	for(int i=0;i<Size;i++)
		image[i] /= max;
}
/* Function that unrolls patches to a [(i*j)x(patchSize*patchSize)] array */
/* returns the patches of every pixel as vectors in an array */
float *unrollPatches(float *paddedImg,int rawImgSize_i,int rawImgSize_j,int patchSize){
	/* Find the dimensions of the patchImg */
	int patchImgSize_i = rawImgSize_i*rawImgSize_j;
	int patchImgSize_j = patchSize * patchSize;
	/* Dimension of paddedImage */
	int padImgSize_j = rawImgSize_j + (patchSize-1);
	/*  malloc memory*/
	float *patchImg = (float *)malloc(patchImgSize_i * patchImgSize_j *sizeof(float));
	/*Iterate each pixel of image */
	for(int i=0;i<rawImgSize_i;i++){
		for(int j=0;j<rawImgSize_j;j++){
			/* Find patch of each pixel */
			for(int p=0;p<patchImgSize_j;p++){
				int row=p/patchSize;
				int col=p%patchSize;
				int patchImg_i = i*rawImgSize_j+ j;
				patchImg[patchImg_i*patchImgSize_j+p] = paddedImg[(i+row)*padImgSize_j+(j+col)]; 
			}
		}
	}
	return patchImg;
}
/* Function that pads the image symmetricaly like octave's padarray(...,[patchsize,patchsize],'symmetric') */
float *padImg(float *data,int rawImgSize_i, int rawImgSize_j, int patchSize,
		int *padImgSize_i,int *padImgSize_j){
	/* Find the Padding space to add */
	int padSpace = (patchSize-1)/2;
	*padImgSize_i = rawImgSize_i + 2*padSpace;
	*padImgSize_j = rawImgSize_j + 2*padSpace;
	/* Allocate memory for padded image */
	float *padImage =  (float *)malloc(*padImgSize_i * *padImgSize_j * sizeof(float));
	/* For each row */
	for (int i=0;i<*padImgSize_i;i++ ){
		for (int j=0;j<*padImgSize_j;j++ ){
			/* For rows between the padding */	
			if(i>=padSpace &&  i<padSpace+rawImgSize_i){
				/* Left Padding (Mirror Data from same row)*/
				if(j<padSpace){
					int row = i - padSpace;
					int col = padSpace - (j+1);
					padImage[i**padImgSize_j+j] = data[row*rawImgSize_j+col];
				}
				/* Right Padding (Mirror Data from same row)*/
				else if(j>=padSpace+rawImgSize_j){
					int row = i - padSpace;
					int col = rawImgSize_j -1 - (j - (rawImgSize_j+padSpace));
					padImage[i**padImgSize_j+j] = data[row*rawImgSize_j+col];
				}
				/* Between the Padding (Copy Data as is) */
				else{
					int row = i - padSpace;
					int col = j - padSpace;
					padImage[i**padImgSize_j+j] = data[row*rawImgSize_j+col];	
				}
			}
			/* For upper rows of the pad  */
			else if(i<padSpace){
				if (j<padSpace){
					int row = padSpace -1 - i;
					int col = padSpace - (j+1);
					padImage[i**padImgSize_j+j] = data[row*rawImgSize_j+col];	
				}
				else if(j>=padSpace+rawImgSize_j){
					int row = padSpace -1 - i;
					int col = rawImgSize_j -1 - (j - (rawImgSize_j+padSpace));
					padImage[i**padImgSize_j+j] = data[row*rawImgSize_j+col];	
				}
				else{
					int row = padSpace -1 - i;
					int col = j-padSpace;
					padImage[i**padImgSize_j+j] = data[row*rawImgSize_j+col];	
				}
			}
			/* For lower rows of the padding */
			else if (i>=padSpace+rawImgSize_i){
				if (j<padSpace){
					int row = rawImgSize_i -1 - (i - (rawImgSize_i+padSpace));
					int col = padSpace - (j+1);
					padImage[i**padImgSize_j+j] = data[row*rawImgSize_j+col];	
				}
				else if(j>=padSpace+rawImgSize_j){
					int row = rawImgSize_i -1 - (i - (rawImgSize_i+padSpace));
					int col = rawImgSize_j -1 - (j - (rawImgSize_j+padSpace));
					padImage[i**padImgSize_j+j] = data[row*rawImgSize_j+col];	
				}
				else{
					int row = rawImgSize_i -1 - (i - (rawImgSize_i+padSpace));
					int col = j-padSpace;
					padImage[i**padImgSize_j+j] = data[row*rawImgSize_j+col];	
				}
			
			}

		}
		
	}
	return padImage;
}

