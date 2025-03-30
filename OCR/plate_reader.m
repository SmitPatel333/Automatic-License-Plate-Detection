clc; clear; close all;

% Load the trained network from the file
loadedData = load('new_license_plate_cnn_model.mat');
dlNet = loadedData.net;  % Extract the dlnetwork object

% Load the new image
newImage = imread('char_6.png');

% Convert to grayscale if necessary
if size(newImage, 3) == 3  % Check if image is RGB
    newImage = rgb2gray(newImage);
end

newImage = imopen(newImage, strel('disk', 1)); % Remove noise

% Resize the image to match the network's input size
newImage = imresize(newImage, [30 27]);

% Add a singleton channel dimension to make it [30, 27, 1]
newImage = reshape(newImage, [30, 27, 1]);

imshow(newImage);

% Convert the image to dlarray (using single precision)
newImageDl = dlarray(single(newImage), 'SSC');  % 'S' for spatial, 'C' for channels

% Make predictions using the trained dlnetwork
predictedLabel = predict(dlNet, newImageDl);

% Display the predicted label
disp('Predicted Label:');
disp(predictedLabel);

[~, predictedClassIdx] = max(predictedLabel, [], 1);  % Get index of highest probability
disp("Predicted Class Index: " + string(predictedClassIdx));

classNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ...
             "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", ...
             "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", ...
             "U", "V", "W", "X", "Y", "Z"];

% Convert index to character
predictedCharacter = classNames(predictedClassIdx);
disp("Predicted Character: " + string(predictedCharacter));