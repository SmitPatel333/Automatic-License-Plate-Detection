digitDatasetPath = fullfile('C:\','Users','Tyler Lee', '.cache', ...
    'kagglehub', 'datasets', 'preatcher', 'standard-ocr-dataset', ...
    'versions', '2', 'data', 'training_data'); 
imds = imageDatastore(digitDatasetPath, "IncludeSubfolders", true, "LabelSource", "foldernames");

img = readimage(imds,1);

% Specify the layers and what they contain
layers = [
    imageInputLayer([30 27 1])

    convolution2dLayer(3,8,Padding="same")
];