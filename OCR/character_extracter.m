clc; clear; close all;

img = imread('licenseplate.png');
gray_img = rgb2gray(img);  % Convert to grayscale

gray_img = imadjust(gray_img);  % Enhance contrast
bw_img = imbinarize(gray_img);  % Convert to binary
bw_img = imcomplement(bw_img);  % Invert (if needed)

bw_img = imopen(bw_img, strel('rectangle', [3, 3]));  % Remove small noise
bw_img = imclearborder(bw_img);  % Remove border noise

cc = bwconncomp(bw_img);
stats = regionprops(cc, 'BoundingBox', 'Area');

charRegions = [];
for i = 1:length(stats)
    bbox = stats(i).BoundingBox;
    aspect_ratio = bbox(3) / bbox(4);  % Width/Height
    if stats(i).Area > 100 && aspect_ratio < 1.5  % Adjust conditions
        charRegions = [charRegions; bbox];
    end
end

for i = 1:size(charRegions, 1)
    char_img = imcrop(gray_img, charRegions(i, :));
    imwrite(char_img, sprintf('char_%d.png', i));  % Save the corrected image
end
