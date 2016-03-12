clear all
close all
clc

%Run scene recognition & localisation on the map for one image.
%not finished, this is just a copy of sceneRecognitionESATDB_testonly.m

%Setup MatConvNet
run matconvnet-1.0-beta16/matlab/vl_setupnn;

% load the pre-trained CNN
net = load('imagenet-vgg-verydeep-16.mat') ;
load('svm.mat')
lastFClayer = 31;

load('newDB.mat','sceneTypes')
uniqueScenes = unique(sceneTypes);
clear sceneTypes



%For each image
testImage = imread(imgetfile);

%normalize image
testImage_ = single(testImage) ; % note: 0-255 range
testImage_ = imresize(testImage_, net.normalization.imageSize(1:2)) ;
testImage_ = testImage_ - net.normalization.averageImage ;

res = vl_simplenn(net, testImage_(:,:,:)) ;
lastFCTesttemp = squeeze(gather(res(lastFClayer+1).x));
lastFCTest = lastFCTesttemp(:);

for i = 1:size(uniqueScenes,1)
    scores(:,i) = W(:,i)'*lastFCTest + B(i) ;
end

[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(testImage) ;
title(sprintf('%s (%d), score %.3f', uniqueScenes{best}, best, bestScore)) ;
