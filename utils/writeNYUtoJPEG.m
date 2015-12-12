clear all
close all
clc

addpath data matconvnet-1.0-beta16 utils data/NYUV2
disp('loading dataset')
load('nyu_depth_v2_labeled.mat')
disp('converting images')
for i = 1:size(images,4)
    imwrite(images(:,:,:,i),fullfile('data/NYUV2',strcat('im',num2str(i),'.jpg')));
    fprintf('.');
end
