clc; clear; close all;

% Load an image.
I = imread("test_plate_7.tif");
grayImg = rgb2gray(I);  % Convert to grayscale (if not already)

threshold = graythresh(grayImg);

binImg = imbinarize(grayImg, threshold); % Binarize the image
binImg = imcomplement(binImg);
binImg = imclearborder(binImg);  % Removes components with fewer than 50 pixels
binImg = imcomplement(binImg);


se = strel('square', 4);

openImg = imerode(binImg, se);
figure
imshow(binImg);
figure
imshow(openImg);

% Perform OCR.
results = ocr(binImg, 'LayoutAnalysis', 'line', 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUOVWXYZ0123456789');

disp(results.Text);