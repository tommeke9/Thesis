% check setup instructions in readme
clear all
close all
clc

%Setup MatConvNet
delete(gcp('nocreate'))
run deps/matconvnet-1.0-beta16/matlab/vl_setupnn;

addpath deps/matconvnet-1.0-beta16

%Object Localisation
n_box_max = 5; % Max amount of boxes to be used for object recognition

%Object Recognition
n_labels_max = 1; %Max amount of recognized objects per box

% load the pre-trained CNN
%net = load('data/cnns/imagenet-matconvnet-vgg-m.mat') ;
net = load('data/cnns/imagenet-matconvnet-vgg-m.mat') ;
if sum(size(net.meta.normalization.averageImage)) == 4
    averageImage(:,:,1) = net.meta.normalization.averageImage(1) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,2) = net.meta.normalization.averageImage(2) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,3) = net.meta.normalization.averageImage(3) * ones(net.meta.normalization.imageSize(1:2));
else
    averageImage = net.meta.normalization.averageImage;
end

%%
testImg = imread(imgetfile);
testObjectLocation = calc_object_locations( n_box_max, testImg );
testObjectRecognition = calc_object_recognition( testImg, testObjectLocation, net, n_labels_max );

bestScoreObject_test = squeeze(testObjectRecognition(1,2,:,:)) ; % bestScoreObject_test(Frame_number,img_Number)
objectNumber_test = squeeze(testObjectRecognition(1,1,:,:)) ; % objectNumber_test(Frame_number,img_Number)
objectName_test = reshape(net.meta.classes.description(objectNumber_test(:)),size(objectNumber_test));
    

testObjectLocation(:,[3 4],1)= testObjectLocation(:,[3 4],1)-testObjectLocation(:,[1 2],1)+1;
%visualization of proposals    

%show first n_box_show boxes in the image
figure
imshow(testImg);
colors = {'blue','black','green','red','cyan','magenta','yellow','white'};

for r=1:size(testObjectLocation,1)
    rectangle('Position', testObjectLocation(r,[1:4],1), 'EdgeColor', colors{r}, 'LineWidth', 5); 
    objectName_temp = strsplit(objectName_test{r},', ');
    objectLabel{r,1} = ['{\color{',colors{r},'}',objectName_temp{1},'(',num2str(bestScoreObject_test(r),'%.2f'),')}; '];
end
title(objectLabel,'FontSize', 20)


