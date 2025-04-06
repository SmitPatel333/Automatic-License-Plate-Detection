scale_factor = 3;

% I2 = imread("Dataset\images\Cars0.png");
I2 = imread("5760055308_33045456f7_h.jpg","jpg");
I1 = imresize(I2, scale_factor);
I = im2gray(I1);
T = graythresh(I);
I = imbinarize(I, T);
I = edge(I, "canny");
SE = strel('square',5*scale_factor);
I = imdilate(I, SE);
I = bwpropfilt(I, "Area", 3, "largest");

% measure perimeter and bounding box for each blob
stats = regionprops(I,{'Area','BoundingBox','perimeter'});
stats = struct2table(stats);
% 1st metric: ratio between perimeter and round length of its bounding box
stats.Metric1 = 2*sum(stats.BoundingBox(:,3:4),2)./stats.Perimeter;
idx1 = abs(1 - stats.Metric1) < 0.2;
% 2nd metric: ratio between blob area and it's bounding box's area
stats.Metric2 = stats.Area./(stats.BoundingBox(:,3).*stats.BoundingBox(:,4));
idx2 = stats.Metric2 > 0.5;

stats.Metric3 = stats.BoundingBox(:,3)./stats.BoundingBox(:,4);
idx3 = stats.Metric3 > 2;
% keep only those which meet both metrics and remove everything else
idx = idx1 & idx2 & idx3;
% stats(~idx,:) = [];

% crop the image using the bounding box
bbox = stats.BoundingBox;
new_I = imcrop(I1, bbox(3,:));
imshow(new_I);
% imshow(I);