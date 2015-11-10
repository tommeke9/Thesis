clear all
clc

addpath data matconvnet-1.0-beta16

%Run setup before! to compile matconvnet
%Variables:
lastFClayer = 36;
RunCNN = 0; %1 = run the CNN, 0 = Load the CNN
RunSVMTraining = 1; %1 = run the SVMtrain, 0 = Load the trained SVM

disp('loading dataset')
load('nyu_depth_v2_labeled.mat')
clear accelData rawDepths rawDepthFilenames
disp('dataset loaded')
[height,width,channels,dbSize] = size(images);

%Setup MatConvNet
run matconvnet-1.0-beta16/matlab/vl_setupnn;

% load the pre-trained CNN
net = load('imagenet-vgg-verydeep-16.mat') ; %TO BE CHANGED

if RunCNN

    % ------------load and preprocess an image---------------------------------
    disp('Normalization')
    % im = imread('data/office.jpg') ;
    for index = 1:dbSize
        im_temp = single(images(:,:,:,index)) ; % note: 0-255 range
        im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
        im_(:,:,:,index) = im_temp - net.normalization.averageImage ;
    end
    clear im_temp 
    disp('Normalization finished')
    %--------------------------------------------------------------------------


    % ---------------------------------Run CNN---------------------------------
    disp('Run CNN')

    for index = 1:dbSize
        %res(:,:,:,index) = vl_simplenn(net, im_(:,:,:,index)) ;
        res = vl_simplenn(net, im_(:,:,:,index)) ;
        lastFC(:,index) = squeeze(gather(res(lastFClayer+1).x));
        fprintf('%d of %d \n',index,dbSize);
        whos('lastFC')
        %disp(num2str(index),'/',num2str(dbSize))
    end
    save('lastFC.mat','lastFC');
    disp('CNN finished')
    %--------------------------------------------------------------------------
else
    load('lastFC.mat');
end


if RunSVMTraining
    %------------------------------Train SVM-----------------------------------
    disp('Train SVM')
    uniqueScenes = unique(sceneTypes);
    [amountOfScenes,~] = size(uniqueScenes);
    for i = 1:amountOfScenes
        thisScene = uniqueScenes(i);
        %disp(thisScene)
        positives = find(strcmp(sceneTypes,thisScene));
        negatives = find(strcmp(sceneTypes,thisScene)==0);
        SVMLabel = zeros(dbSize,1);
        SVMLabel(negatives) = -1;
        SVMLabel(positives) = 1;

        [X,Y,INFO] = vl_svmtrain(lastFC,SVMLabel,0.1,'verbose');
        if i==1
            W = X;
            B = Y;
        else
            W = [W,X];
            B = [B,Y];
        end
    end
    save('svm.mat','W','B', 'uniqueScenes');
    disp('Training finished')
    %--------------------------------------------------------------------------
else
    load('svm.mat');
end

%---------------------------Validate---------------------------------------
testImage = imread('data/bookstore.jpg');
testImage_ = single(testImage) ; % note: 0-255 range
testImage_ = imresize(testImage_, net.normalization.imageSize(1:2)) ;
testImage_ = testImage_ - net.normalization.averageImage ;
res = vl_simplenn(net, testImage_(:,:,:)) ;
lastFCTest = squeeze(gather(res(lastFClayer+1).x));

for i = 1:amountOfScenes
    scores(:,i) = W(:,i).*lastFCTest + B(i) ; %changed the * to . 
end
scorePerScene = sum(scores)+abs(min(sum(scores)));
[bestScore, best] = max(scorePerScene) ;
figure(1) ; clf ; imagesc(testImage) ;
title(sprintf('%s (%d), score %.3f', uniqueScenes{best}, best, bestScore)) ;
%--------------------------------------------------------------------------