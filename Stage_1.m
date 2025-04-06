scale_factor = 3; % scale at which the image and structuring element are upsampled

% Read the image and resize it
I1 = imread("Dataset\images\Cars40.png");
I1 = imresize(I1, scale_factor);

% Record image size for later use
[height, width, depth] = size(I1);

% Process the image to extract edges
I = im2gray(I1);
I = imbinarize(I, .5);
I = edge(I, "canny");

% Dilate image to create connected regions
SE = strel('square',1*scale_factor);
for i = 1:6
    I = imdilate(I, SE);
end

% Keep only the 5 largest regions by area
I = bwpropfilt(I, "Area", 10, "largest");

% record the perimeter, area, and bounding box of each region
stats = regionprops(I, {'Area','BoundingBox','perimeter'});
stats = struct2table(stats);

% Calculate various metrics and construct vectors stating which regions
% fulfill each metric
% 1st metric: ratio between perimeters of each region and its bounding box
stats.Metric1 = 2*sum(stats.BoundingBox(:,3:4),2)./stats.Perimeter;
idx1 = abs(1 - stats.Metric1) < 0.2;

% 2nd metric: ratio between areas of each region and its bounding box
stats.Metric2 = stats.Area./(stats.BoundingBox(:,3).*stats.BoundingBox(:,4));
idx2 = stats.Metric2 > 0.3;

% 3rd metric: ratio of bounding box length and width, it should be a rectangle
stats.Metric3 = stats.BoundingBox(:,3)./stats.BoundingBox(:,4);
idx3 = stats.Metric3 > 1.5;

% 4th metric: the starting point of the bounding box in the x direction must 
% be within the middle quarters of the image width
stats.Metric4 = stats.BoundingBox(:,1);
idx4 = stats.Metric4 > width/4 & stats.Metric4 < width*3/4;

% 5th metric: the starting point of the bounding box in the y direction must 
% be within the middle quarters of the image height
stats.Metric5 = stats.BoundingBox(:,2);
idx5 = stats.Metric5 > height/4 & stats.Metric5 < height*3/4;

% 6th metric: ratio between the region area and the total area of the image
stats.Metric6 = stats.Area./(width*height);
idx6 = stats.Metric6 > .01;

% keep only the region(s) which meet all metrics and remove everything else
% there should only be one
idx = idx1 & idx2 & idx3 & idx4 & idx5 & idx6;
stats(~idx,:) = [];

% display the image with the bounding box overlaid
bbox = stats.BoundingBox;
bbox_I = insertShape(I1,"rectangle",bbox,"LineWidth",6);
imshow(bbox_I);

% crop and save the bounded region
new_I = imcrop(I1, bbox);
% imwrite(new_I, "cropped_license_plate.tif");
% imshow(new_I); % uncomment if you want to see the cropped area