clear all
close all
clc

addpath data deps/matconvnet-1.0-beta16 utils
disp('loading cnn')
%load('imagenet-vgg-verydeep-16.mat')
load('imagenet-caffe-ref.mat')
disp('cnn loaded')
for i = 1:size(layers,2)
    types(i,1) = cellstr(layers{i}.type);
    names(i,1) = cellstr(layers{i}.name);
end