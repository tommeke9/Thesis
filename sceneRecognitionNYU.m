clear all
clc

addpath data matconvnet-1.0-beta16

%Run setup before! to compile matconvnet
%Variables:
lastFClayer = 36;

disp('loading dataset')
load('nyu_depth_v2_labeled.mat')
disp('dataset loaded')
[height,width,channels,dbSize] = size(images);

%Setup MatConvNet
run matconvnet-1.0-beta16/matlab/vl_setupnn;

% load the pre-trained CNN
net = load('imagenet-vgg-verydeep-16.mat') ; %TO BE CHANGED

% ------------load and preprocess an image---------------------------------
disp('Normalization')
% im = imread('data/office.jpg') ;
for index = 1:dbSize
    im(:,:,:,index) = images(:,:,:,index);
    im_temp = single(im(:,:,:,index)) ; % note: 0-255 range
    im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
    im_(:,:,:,index) = im_temp - net.normalization.averageImage ;
end
disp('Normalization finished')
%--------------------------------------------------------------------------


% ---------------------------------Run CNN---------------------------------
disp('Run CNN')

for index = 1:dbSize
    res(:,:,:,index) = vl_simplenn(net, im_(:,:,:,index)) ;
    fprintf('%d of %d \n',index,dbSize);
    whos('res')
    %disp(num2str(index),'/',num2str(dbSize))
end
disp('CNN finished')
%--------------------------------------------------------------------------


%------------------------------Train SVM-----------------------------------
disp('Train SVM')
uniqueScenes = unique(sceneTypes);
[amountOfScenes,~] = size(uniqueScenes);
for i = 1:amountOfScenes
    thisScene = uniqueScenes(i);
    Disp('now training: ', thisScene)
    positives = find(strcmp(sceneTypes,thisScene));
    negatives = find(strcmp(sceneTypes,thisScene)==0);
    SVMLabel = zeros(dbSize,1);
    SVMLabel(negatives) = -1;
    SVMLabel(positives) = 1;
    
    [W,B,INFO] = vl_svmtrain(res(:,:,:,:),SVMLabel,0.1,'conserveMemory', true);
end
disp('Training finished')
%--------------------------------------------------------------------------


% % run the CNN
% res = vl_simplenn(net, im_) ;
% 
% % show the classification result
% scores = squeeze(gather(res(lastFClayer+1).x)) ;
% [bestScore, best] = max(scores) ;
% figure(1) ; clf ; imagesc(im) ;
% title(sprintf('%s (%d), score %.3f',...
% net.classes.description{best}, best, bestScore)) ;