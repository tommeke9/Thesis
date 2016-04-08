clear all
close all
clc

addpath data deps/matconvnet-1.0-beta16 utils
disp('loading cnn')
load('data/cnns/imagenet-vgg-verydeep-16.mat')
%load('data/cnns/imagenet-caffe-ref.mat')
%load('data/cnns/imagenet-matconvnet-vgg-m.mat')
disp('cnn loaded')
for i = 1:size(layers,2)
    types(i,1) = cellstr(layers{i}.type);
    names(i,1) = cellstr(layers{i}.name);
end