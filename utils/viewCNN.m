clear all
close all
clc

TrainingCoordinates = makeTrainingCoordinates();
addpath data deps/matconvnet-1.0-beta16 data/ESAT-DB

%Run setup before! to compile matconvnet
%%
%------------------------VARIABLES-----------------------------------------

lastFClayer = 13;
net = load('data/cnns/imagenet-matconvnet-vgg-m.mat') ;
if sum(size(net.meta.normalization.averageImage)) == 4
    averageImage(:,:,1) = net.meta.normalization.averageImage(1) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,2) = net.meta.normalization.averageImage(2) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,3) = net.meta.normalization.averageImage(3) * ones(net.meta.normalization.imageSize(1:2));
else
    averageImage = net.meta.normalization.averageImage;
end

delete(gcp('nocreate'))
run deps/matconvnet-1.0-beta16/matlab/vl_setupnn;

cam = webcam(1);
figure
for loopIndex = 1:500
    tic
    testImg = snapshot(cam);

    %normalize
    im_temp = single(testImg) ; % note: 0-255 range
    im_temp = imresize(im_temp, net.meta.normalization.imageSize(1:2)) ;
    testImgNorm = im_temp - averageImage ;
    
    %Run cnn
    res = vl_simplenn(net, testImgNorm) ;
    lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
    lastFCtest = lastFCtemp(:);
    toc

    subplot(2,3,2)
    imshow(testImg);

    subplot(2,3,4:6)
    plot(lastFCtest)
    title('Output of layer')
    
end