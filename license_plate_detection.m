car = imread("Dataset\images\Cars11.png");
[bbox, cropped] = locatePlate(car);
text = readPlate(cropped);

imshow(bbox);
disp(text);

function [bbox_I, cropped_I] = locatePlate(car_image)
    scale_factor = 3; % scale at which the image and structuring element are upsampled
    
    % Resize the image
    original_I = imresize(car_image, scale_factor);
    
    % Record image size for later use
    [height, width, depth] = size(original_I);
    
    % Process the image to extract edges
    I = im2gray(original_I);
    I = imbinarize(I, .5);
    I = edge(I, "canny");
    
    % Dilate image to create connected regions from edges
    SE = strel('square', 1*scale_factor);
    for i = 1:6
        I = imdilate(I, SE);
    end
    
    % Keep only the 10 largest regions by area
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
    idx2 = stats.Metric2 > 0.5;
    
    % 3rd metric: ratio of bounding box length and width, it should be a rectangle
    stats.Metric3 = stats.BoundingBox(:,3)./stats.BoundingBox(:,4);
    idx3 = stats.Metric3 > 1.5 & stats.Metric3 < 5;
    
    % 4th metric: the starting point of the bounding box in the x direction must 
    % be within the middle fifths of the image width
    stats.Metric4 = stats.BoundingBox(:,1);
    idx4 = stats.Metric4 > width/5 & stats.Metric4 < width*4/5;
    
    % 5th metric: the starting point of the bounding box in the y direction must 
    % be below the top quarter of the image
    stats.Metric5 = stats.BoundingBox(:,2);
    idx5 = stats.Metric5 > height/4;
    
    % 6th metric: ratio between the region area and the total area of the image
    stats.Metric6 = stats.Area./(width*height);
    idx6 = stats.Metric6 > .005;
    
    % keep only the region(s) which meet all metrics and remove everything else
    % there should only be one
    idx = idx1 & idx2 & idx3 & idx4 & idx5 & idx6;
    stats(~idx,:) = [];
    
    % return the image with the bounding box overlaid
    bbox = stats.BoundingBox;
    bbox_I = insertShape(original_I,"rectangle",bbox,"LineWidth",6);
    
    % crop and return the bounded region
    cropped_I = imcrop(original_I, bbox);
end

function plate_text = readPlate(plateImage)
    grayImg = rgb2gray(plateImage);  % Convert to grayscale (if not already)
    
    threshold = graythresh(grayImg);
    binImg = imbinarize(grayImg, threshold); % Binarize the image
 
    binImg = imcomplement(binImg);
    binImg = imclearborder(binImg);  % Removes components with fewer than 50 pixels
    binImg = imcomplement(binImg);
        
    se = strel('square', 3);
    openImg = imerode(binImg, se);

    % Perform OCR.
    results = ocr(openImg, 'TextLayout', 'line', 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUOVWXYZ0123456789');

    plate_text = results.Text;
end