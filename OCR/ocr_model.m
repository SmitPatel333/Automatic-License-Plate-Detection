digitDatasetPath = fullfile('C:\','Users','Tyler Lee', '.cache', ...
    'kagglehub', 'datasets', 'preatcher', 'standard-ocr-dataset', ...
    'versions', '2', 'data', 'training_data'); 
imds = imageDatastore(digitDatasetPath, "IncludeSubfolders", true, "LabelSource", "foldernames");

img = readimage(imds,1);

inputSize = [30 27 1]; % CNN input
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXScale', [1 1.2], ...
    'RandYScale', [1 1.2]);

augimds = augmentedImageDatastore(inputSize, imds, 'DataAugmentation', imageAugmenter);

% Specify the layers and what they contain
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
	
    maxPooling2dLayer(2,Stride=2)
	
    convolution2dLayer(3,64,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,128,Padding="same")
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(36)
    softmaxLayer
    classificationLayer

];

options = trainingOptions("sgdm", ...
    InitialLearnRate=0.1, ...
    MaxEpochs=4, ...
    Plots="training-progress");    

net = trainNetwork(augimds,layers,options);

save('license_plate_cnn_model.mat', 'net')