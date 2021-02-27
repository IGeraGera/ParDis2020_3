
void normalizeImage(float *image,int Size_i,int Size_j);
float *unrollPatches(float *paddedImg,int rawImgSize_i,int rawImgSize_j,int patchSize);
float *padImg(float *data,int rawImgSize_i, int rawImgSize_j, int patchSize,
		int *padImgSize_i,int *padImgSize_j);
