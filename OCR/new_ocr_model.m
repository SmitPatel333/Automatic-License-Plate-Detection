clc; clear; close all;

digitDatasetPath = fullfile('C:\','Users','Tyler Lee', '.cache', ...
    'kagglehub', 'datasets', 'preatcher', 'standard-ocr-dataset', ...
    'versions', '2', 'data', 'training_data'); 
validationDatasetPath = fullfile('C:\','Users','Tyler Lee', '.cache', ...
    'kagglehub', 'datasets', 'preatcher', 'standard-ocr-dataset', ...
    'versions', '2', 'data', 'testing_data'); 

imds = imageDatastore(digitDatasetPath, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

% classNames = categories(imds.Labels);
% 
% disp("Class Labels:");
% disp(classNames);

imdsValidation = imageDatastore(validationDatasetPath, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

targetSize = [30 27];

% Set the custom ReadFcn to resize images when they are loaded
imds.ReadFcn = @(filename) imresize(imread(filename), targetSize);

% Verify the new size of one image
img = readimage(imds, 1);
disp(size(img));  % Should show [30 27 1] for grayscale images

classNames = categories(imds.Labels);
labelCount = countEachLabel(imds)

layers = [
    imageInputLayer([30 27 1])

    convolution2dLayer(3,8,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,16,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(36)
    softmaxLayer];

options = trainingOptions("sgdm", ...
    InitialLearnRate=0.01, ...
    MaxEpochs=4, ...
    Shuffle="every-epoch", ...
    ValidationData=imds, ...
    ValidationFrequency=30, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

net = trainnet(imds,layers,"crossentropy",options);

save('new_license_plate_cnn_model.mat', 'net')

% scores = minibatchpredict(net,imdsValidation);
% YValidation = scores2label(scores,classNames);
% 
% TValidation = imdsValidation.Labels;
% accuracy = mean(YValidation == TValidation)

