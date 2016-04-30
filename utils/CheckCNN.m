clear all
close all
clc

addpath data deps/matconvnet-1.0-beta16

%Run setup before! to compile matconvnet
%%
%------------------------VARIABLES-----------------------------------------
imDB = 3;
lastFClayer = 13;
net = load('data/cnns/imagenet-matconvnet-vgg-m.mat') ;



if sum(size(net.meta.normalization.averageImage)) == 4
    averageImage(:,:,1) = net.meta.normalization.averageImage(1) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,2) = net.meta.normalization.averageImage(2) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,3) = net.meta.normalization.averageImage(3) * ones(net.meta.normalization.imageSize(1:2));
else
    averageImage = net.meta.normalization.averageImage;
end


switch imDB
    case 1
        T = load('data/thuis/mat/thuis1.mat');
        images = T.img; %different objects in the image
        clear T
    case 2
        T = load('data/thuis/mat/thuis1.mat');
        images = T.img(:,:,:,[6,40,65,76,108,131,194,301,302,303,304,305]); %selection of key-images
        clear T
    case 3
        T = load('data/thuis/mat/thuis2.mat');
        images = T.img; %different viewpoints
        clear T
end
disp('DB loaded')

delete(gcp('nocreate'))
run deps/matconvnet-1.0-beta16/matlab/vl_setupnn;

figure
for index = 1:size(images,4)
    testImg = images(:,:,:,index);

    %normalize
    im_temp = single(testImg) ; % note: 0-255 range
    im_temp = imresize(im_temp, net.meta.normalization.imageSize(1:2)) ;
    testImgNorm = im_temp - averageImage ;
    
    %Run cnn
    res = vl_simplenn(net, testImgNorm) ;
    lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
    lastFCtest = lastFCtemp(:);
    
    if index == 1
        LayerReference = lastFCtest;
    end

    subplot(3,2,1)
    imshow(images(:,:,:,1));
    title('Reference image')

    subplot(3,2,2)
    imshow(testImg);
    title('Current image')
    
    subplot(3,2,3:4)
    plot(lastFCtest)
    axis([0 size(lastFCtest,1) 0 5])
    title('Output of layer')
    
    subplot(3,2,5:6)
    plot((lastFCtest-LayerReference).^2)
    title(['Change to reference: ',num2str(norm(lastFCtest-LayerReference))])
    axis([0 size(lastFCtest,1) 0 10])
    drawnow
end