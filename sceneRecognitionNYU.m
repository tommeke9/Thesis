clear all
clc

addpath data matconvnet-1.0-beta16

%Run setup before! to compile matconvnet
%Variables:
lastFClayer = 36;
RunCNN = 0; %1 = run the CNN, 0 = Load the CNN
RunSVMTraining = 0; %1 = run the SVMtrain, 0 = Load the trained SVM
AmountTestImagesPerClass = 3; %Amount of Validation Images per class

disp('loading dataset')
load('nyu_depth_v2_labeled.mat')
clear accelData rawDepths rawDepthFilenames
uniqueScenes = unique(sceneTypes);
[amountOfScenes,~] = size(uniqueScenes);
disp('dataset loaded')
[height,width,channels,dbSize] = size(images);

%Split the DB into Training and Validation (linking to original DB)
validationIndex = 1;
trainingIndex = 1;
for i = 1:amountOfScenes
        thisScene = uniqueScenes(i);
        locationOfScene = find(strcmp(sceneTypes,thisScene));
        for y = 1:size(locationOfScene,1)
            if y<=AmountTestImagesPerClass
                validationDB(validationIndex) = locationOfScene(y);
                validationIndex = validationIndex + 1;
                %validationDB(:,:,:,validationIndex) = images(:,:,:,locationOfScene(y));
            else
                trainingDB(trainingIndex) = locationOfScene(y);
                trainingIndex = trainingIndex + 1; 
                %trainingDB(:,:,:,trainingIndex) = images(:,:,:,locationOfScene(y));
            end
        end
        if size(locationOfScene,1)<=AmountTestImagesPerClass
            fprintf('No trainingdata for: %s\n',thisScene{:});
        end  
end
% Define the training subset of the DB
trainingDBSize = size(trainingDB,2);
validationDBSize = size(validationDB,2);

%Setup MatConvNet
run matconvnet-1.0-beta16/matlab/vl_setupnn;

% load the pre-trained CNN
net = load('imagenet-vgg-verydeep-16.mat') ; %TO BE CHANGED TO VGG Places2

if RunCNN

    
    % ------------load and preprocess the images---------------------------------
    disp('Normalization')
    % im = imread('data/office.jpg') ;
    for index = 1:trainingDBSize
        im_temp = single(images(:,:,:,trainingDB(index))) ; % note: 0-255 range
        im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
        im_(:,:,:,index) = im_temp - net.normalization.averageImage ;
    end
    clear im_temp 
    disp('Normalization finished')
    %--------------------------------------------------------------------------


    % ---------------------------------Run CNN---------------------------------
    disp('Run CNN')
    for index = 1:trainingDBSize
        %res(:,:,:,index) = vl_simplenn(net, im_(:,:,:,index)) ;
        res = vl_simplenn(net, im_(:,:,:,index)) ;
        lastFC(:,index) = squeeze(gather(res(lastFClayer+1).x));
        fprintf('%d of %d \n',index,trainingDBSize);
        %whos('lastFC')
        %disp(num2str(index),'/',num2str(dbSize))
    end
    save('lastFC.mat','lastFC');
    clear im_ im_temp
    disp('CNN finished')
    %--------------------------------------------------------------------------
else
    load('lastFC.mat');
end


if RunSVMTraining
    %------------------------------Train SVM-----------------------------------
    disp('Train SVM')
    %New database of validate images 
    for index = 1:trainingDBSize
        sceneTrainTypes(index) = sceneTypes(trainingDB(index));
    end
    
    for i = 1:amountOfScenes
        thisScene = uniqueScenes(i);
        %disp(thisScene)
        positives = find(strcmp(sceneTrainTypes,thisScene));
        negatives = find(strcmp(sceneTrainTypes,thisScene)==0);
        SVMLabel = zeros(trainingDBSize,1);
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
    save('svm.mat','W','B');
    disp('Training finished')
    %--------------------------------------------------------------------------
else
    load('svm.mat');
end

%---------------------------Validate (ROC)---------------------------------------
disp('Start validation')
correct = 0;
for index = 1:validationDBSize
    %Normalize
    im_temp = single(images(:,:,:,validationDB(index))) ; % note: 0-255 range
    im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
    im_(:,:,:,index) = im_temp - net.normalization.averageImage ;
    %Run CNN
    res = vl_simplenn(net, im_(:,:,:,index)) ; 
    lastTrainingFC = squeeze(gather(res(lastFClayer+1).x));
    
    for i = 1:amountOfScenes
        scores(:,i) = W(:,i)'*lastTrainingFC + B(i) ;
    end
    
    [bestScore(index), best(index)] = max(scores) ;
    
    %Check against given scene
    if strcmp(uniqueScenes{best(index)}, sceneTypes(validationDB(index)))
        correct = correct + 1;
        fprintf('CORRECT: %s \n',uniqueScenes{best(index)});
    else
        fprintf('Wrong: %s but correct is %s \n',uniqueScenes{best(index)},sceneTypes{validationDB(index)});
    end
    
    fprintf('%d of %d \n',index,validationDBSize);
end

disp('Validation finished')
fprintf('Result: %d out of %d are correct\n',correct,validationDBSize);
%--------------------------------------------------------------------------


% %---------------------------MANUAL Validate---------------------------------------
% testImage = imread('data/bookstore.jpg');
% testImage_ = single(testImage) ; % note: 0-255 range
% testImage_ = imresize(testImage_, net.normalization.imageSize(1:2)) ;
% testImage_ = testImage_ - net.normalization.averageImage ;
% res = vl_simplenn(net, testImage_(:,:,:)) ;
% lastFCTest = squeeze(gather(res(lastFClayer+1).x));
% 
% for i = 1:amountOfScenes
%     scores(:,i) = W(:,i)'*lastFCTest + B(i) ; %changed the * to . 
% end
% %scorePerScene = sum(scores)+abs(min(sum(scores)));
% [bestScore, best] = max(scores) ;
% figure(1) ; clf ; imagesc(testImage) ;
% title(sprintf('%s (%d), score %.3f', uniqueScenes{best}, best, bestScore)) ;
% %--------------------------------------------------------------------------