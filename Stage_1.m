I1 = imread("Dataset\images\Cars11.png");
% I1 = imread("otsu.tif");
I = im2gray(I1);
T = graythresh(I);
I = imbinarize(I, .5);
I = edge(I, "canny");
SE = strel('square',5);
I = imdilate(I, SE);
I = bwpropfilt(I, "Area", 5, "largest");

% measure perimeter and bounding box for each blob
stats = regionprops(I,{'Area','BoundingBox','perimeter'});
stats = struct2table(stats);
% 1st metric: ratio between perimeter and round length of its bounding box
stats.Metric1 = 2*sum(stats.BoundingBox(:,3:4),2)./stats.Perimeter;
idx1 = abs(1 - stats.Metric1) < 0.1;
% 2nd metric: ratio between blob area and it's bounding box's area
stats.Metric2 = stats.Area./(stats.BoundingBox(:,3).*stats.BoundingBox(:,4));
idx2 = stats.Metric2 > 0.8;
% keep only those which meet both metrics and remove everything else
idx = idx1 & idx2;
stats(~idx,:) = [];

% crop the image using the bounding box
% bbox = stats.BoundingBox;
% new_I = imcrop(I1, bbox);
% imshow(new_I);
imshow(I);