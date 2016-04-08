clear all
close all
clc


%Setup MatConvNet
run deps/matconvnet-1.0-beta16/matlab/vl_setupnn;

% load the pre-trained CNN
net = load('data/cnns/imagenet-vgg-verydeep-16.mat') ; %TO BE CHANGED TO VGG Places2

load('svm.mat')
load('scenes.mat') %To be deleted...
lastFClayer = 31;

testImage = imread(imgetfile);
testImage_ = single(testImage) ; % note: 0-255 range
testImage_ = imresize(testImage_, net.normalization.imageSize(1:2)) ;
testImage_ = testImage_ - net.normalization.averageImage ;

res = vl_simplenn(net, testImage_(:,:,:)) ;
lastFCTesttemp = squeeze(gather(res(lastFClayer+1).x));
lastFCTest = lastFCTesttemp(:);

for i = 1:size(uniqueScenes,1)
    scores(:,i) = W(:,i)'*lastFCTest + B(i) ; %changed the * to . 
end

[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(testImage) ;
title(sprintf('%s (%d), score %.3f', uniqueScenes{best}, best, bestScore)) ;
