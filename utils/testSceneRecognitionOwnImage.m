clear all
close all
clc


%Setup MatConvNet
run deps/matconvnet-1.0-beta16/matlab/vl_setupnn;

% load the pre-trained CNN
net = load('data/cnns/imagenet-matconvnet-vgg-m.mat') ;
if sum(size(net.meta.normalization.averageImage)) == 4
    averageImage(:,:,1) = net.meta.normalization.averageImage(1) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,2) = net.meta.normalization.averageImage(2) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,3) = net.meta.normalization.averageImage(3) * ones(net.meta.normalization.imageSize(1:2));
else
    averageImage = net.meta.normalization.averageImage;
end

load('data/scene/ESATsvm.mat')
lastFClayer = 13;

%%
testImage = imread(imgetfile);
testImage_ = single(testImage) ; % note: 0-255 range
testImage_ = imresize(testImage_, net.meta.normalization.imageSize(1:2)) ;
testImage_ = testImage_ - averageImage ;

res = vl_simplenn(net, testImage_(:,:,:)) ;
lastFCTesttemp = squeeze(gather(res(lastFClayer+1).x));
lastFCTest = lastFCTesttemp(:);

for i = 1:size(uniqueScenes,1)
    scores(:,i) = W(:,i)'*lastFCTest + B(i) ;
end

[bestScore, best] = max(scores) ;
figure(1);
clf;
imagesc(testImage);
title(sprintf('%s, score %.3f', uniqueScenes{best}, bestScore),'Interpreter','none','FontSize', 20);
