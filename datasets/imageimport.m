# Crop the image to size 
pkg load image
normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

# Select Size
size = 64;
size--;
# Import Image
rgbimg = imread("house.jpg");
img=double(rgb2gray(rgbimg))/255;
normImg(img);
img = img(250:250+size,300:300+size);
# Print image
figure("Name","Original");imagesc(img);axis image;colormap gray;
# Add Noise
noiseParams = {'gaussian', ...
                 0,...
                 0.001};
noiseimg = imnoise( img, noiseParams{:} );
# Print image
figure("Name","Noisy");imagesc(noiseimg);axis image;colormap gray;

