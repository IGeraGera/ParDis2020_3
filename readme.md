# ParDis2020\_3

This repo contains the source code for the 3rd assignment of Parallel and Distributed Systems which is a CUDA implementation in C for Non Local Means image denoising algortihm.

Antoni Buades, Bartomeu Coll, and J­M Morel.  A non­local algorithm for image denoising.  In2005 IEEEComputer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), volume 2, pages 60–65.IEEE, 2005

## Structure

datasets : contains the .m script that crops and preprocesses a .jpg image and the cropped images used in .csv form.

results : contains the results from executing the code in a uni hpc.

inc : header files

## Use 

make mainMemoryLimit test

More in Makefile ...

The result is saved in the directory of the input file and can be read with matlab/octave. The execution time is printed in stdout. 

For the assignment we were instructed to denoise square images 64x64, 128x128 and 256x256 with patch size 3x3, 5x5 and 7x7.

Preprocess in importimage.m

+  Import image
+  Convert to grayscale
+  normalize to [0,1]
+  min-max(almost) normalization
