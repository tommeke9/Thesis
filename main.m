clear all
clc
%TO BE USED TO CALL scneRecognition.m AND objectRecognition.m
% 
% addpath data matconvnet-1.0-beta16
% 
% %Run setup before! to compile matconvnet
% 
% %Setup MatConvNet
% run matconvnet-1.0-beta16/matlab/vl_setupnn;
% 
% % load the pre-trained CNN
% net = load('imagenet-vgg-verydeep-16.mat') ;
% %net = load('imagenet-googlenet-dag.mat') ;
% 
% % load and preprocess an image
% im = imread('data/office.jpg') ;
% im_ = single(im) ; % note: 0-255 range
% im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
% im_ = im_ - net.normalization.averageImage ;
% 
% % run the CNN
% res = vl_simplenn(net, im_) ;
% 
% % show the classification result
% scores = squeeze(gather(res(end).x)) ;
% [bestScore, best] = max(scores) ;
% figure(1) ; clf ; imagesc(im) ;
% title(sprintf('%s (%d), score %.3f',...
% net.classes.description{best}, best, bestScore)) ;