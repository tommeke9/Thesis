clear all
close all
clc

disp('loading ESAT DB')
T = load('test.mat');
testImg = T.img;
clear T
T = load('training.mat');
trainingImg = T.img;
clear T
disp('DB loaded')

% Define the sizes of the DB
trainingDBSize = size(trainingImg,4);
testDBSize = size(testImg,4);


%Normalization not necessary! ==> logbook 28/02/2016
% disp('Normalization')
% run matconvnet-1.0-beta16/matlab/vl_setupnn;
% net = load('imagenet-vgg-verydeep-16.mat') ; %TO BE CHANGED TO VGG Places2
%     for index = 1:trainingDBSize
%         im_temp = single(trainingImg(:,:,:,index)) ; % note: 0-255 range
%         im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
%         trainingImgNorm(:,:,:,index) = im_temp - net.normalization.averageImage ;
%     end
%     clear im_temp index
%     for index = 1:testDBSize
%         im_temp = single(testImg(:,:,:,index)) ; % note: 0-255 range
%         im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
%         testImgNorm(:,:,:,index) = im_temp - net.normalization.averageImage ;
%     end
%     clear im_temp index
%     disp('Normalization finished')



for index = 1:trainingDBSize
[BW(:,:,index),threshOut(index)] = edge(rgb2gray(trainingImg(:,:,:,index)));
end

for index = 1:testDBSize
[BWtest(:,:,index),threshOutTest(index)] = edge(rgb2gray(testImg(:,:,:,index)));
end

figure;
histogram(threshOut)
title('Tresholds Training DB')
xlabel('Value of treshold')
ylabel('Amount of images')

figure;
histogram(threshOutTest)
title('Tresholds Test DB')
xlabel('Value of treshold')
ylabel('Amount of images')

figure;
for i=1:trainingDBSize
    if threshOut(i) < 0.05
        imshow(trainingImg(:,:,:,i))
        pause(.005);
    end
end
