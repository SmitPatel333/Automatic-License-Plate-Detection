% I = imread("Dataset\images\Cars24.png");
I1 = imread("otsu.tif");
I = im2gray(I1);
T = graythresh(I);
I = imbinarize(I, .5);
I = edge(I, "canny");
SE = strel('diamond',9);
I = imdilate(I, SE);

I = bwpropfilt(I, "Area", [10000 50000]);
CC = bwconncomp(I);
stats = regionprops(CC, "Area", "BoundingBox");
bbox = stats.BoundingBox;

new_I = imcrop(I1, bbox);

imshow(new_I);