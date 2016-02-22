clear all
clc

addpath data matconvnet-1.0-beta16 data/ESAT-DB

%Run setup before! to compile matconvnet
%Variables:
lastFClayer = 31;
RunCNN = 0; %1 = run the CNN, 0 = Load the CNN
RunConf = 0; %1 = recalc the Conf. matrix, 0 = Load the Conf. Matrix

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

%Setup MatConvNet
run matconvnet-1.0-beta16/matlab/vl_setupnn;

if RunCNN
    % load the pre-trained CNN
    net = load('imagenet-vgg-verydeep-16.mat') ; %TO BE CHANGED TO VGG Places2



    
    % ------------load and preprocess the images---------------------------------
    disp('Normalization')
    for index = 1:trainingDBSize
        im_temp = single(trainingImg(:,:,:,index)) ; % note: 0-255 range
        im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
        trainingImgNorm(:,:,:,index) = im_temp - net.normalization.averageImage ;
    end
    clear im_temp index
    for index = 1:testDBSize
        im_temp = single(testImg(:,:,:,index)) ; % note: 0-255 range
        im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
        testImgNorm(:,:,:,index) = im_temp - net.normalization.averageImage ;
    end
    clear im_temp index
    disp('Normalization finished')
    %--------------------------------------------------------------------------


    % ---------------------------------Run CNN---------------------------------
    disp('Run CNN')
    for index = 1:trainingDBSize
        res = vl_simplenn(net, trainingImgNorm(:,:,:,index)) ;
        
        lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
        lastFCtraining(:,index) = lastFCtemp(:);
        
        if rem(index,100)==0
            fprintf('training %d ~ %d of %d \n',index-99,index,trainingDBSize);
        end
    end
    save('data/lastFCesatDB.mat','lastFCtraining');
    clear lastFCtemp index res
    
    %Same for testDB
    for index = 1:testDBSize
        res = vl_simplenn(net, testImgNorm(:,:,:,index)) ;
        
        lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
        lastFCtest(:,index) = lastFCtemp(:);
        
        if rem(index,100)==0
            fprintf('test %d ~ %d of %d \n',index-99,index,trainingDBSize);
        end
    end
    save('data/lastFCesatDB.mat','lastFCtest');
    clear lastFCtemp res
    disp('CNN finished')
    %--------------------------------------------------------------------------
else
    disp('CNN not recalculated')
    load('lastFCesatDB.mat');
end



%---------------------------Test (Confusion)---------------------------------------
if RunConf
    disp('Start tests')
    confusionMatrix = zeros(trainingDBSize);
    for index = 1:testDBSize
        for i = 1:trainingDBSize
            confusionMatrix(i,index) = norm(lastFCtest(:,index)-lastFCtraining(:,i));
        end
        if rem(index,100)==0
                fprintf('Confusion Calc. %d ~ %d of %d \n',index-99,index,testDBSize);
        end
    end
    save('data/confMatrix.mat','confusionMatrix');
else
    disp('ConfusionMatrix not recalculated')
    load('confMatrix.mat');
end

figure;
imagesc(confusionMatrix)

%--------------------------------------------------------------------------

%-------------------------------Select Lowest difference--------------------------
disp('Search lowest difference')
for index = 1:testDBSize
    [~,Result(index)] = min(confusionMatrix(:,index));
    if rem(index,100)==0
            fprintf('%d ~ %d of %d \n',index-99,index,testDBSize);
    end
end