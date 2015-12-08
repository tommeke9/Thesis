clear all
clc

addpath data matconvnet-1.0-beta16

%Run setup before! to compile matconvnet
%Variables:
lastFClayer = 36;
RunCNN = 1; %1 = run the CNN, 0 = Load the CNN
RunSVMTraining = 1; %1 = run the SVMtrain, 0 = Load the trained SVM

%C = [0.01,0.1:0.1:1.5,2:2:100,100:200:1000,1000:50000:1000000]; %All C's to Validate
%C = [0.01,0.1:0.1:1.5];
C = [0.001,0.01,0.1:0.2:1.5,2:2:100,100:200:1000,1000,1000000];

ValidationPercentage = 15;
TestPercentage = 15;
TrainPercentage = 70;
ClassTreshold = 20; %below this number of images in class not useful

if ValidationPercentage+TestPercentage+TrainPercentage~=100
    disp('Check Test, Train, and Validation percentages')
    return
end

disp('loading dataset')
load('nyu_depth_v2_labeled.mat')
clear accelData rawDepths rawDepthFilenames
uniqueScenes = unique(sceneTypes);
[amountOfScenes,~] = size(uniqueScenes);
disp('dataset loaded')
[height,width,channels,dbSize] = size(images);

%Split the DB into Training, Test and Validation (linking to original DB)
testIndex = 1;
trainingIndex = 1;
ValIndex = 1;
fprintf('Not enough data for: ');
for i = 1:amountOfScenes
        thisScene = uniqueScenes(i);
        locationOfScene = find(strcmp(sceneTypes,thisScene));
        for y = 1:size(locationOfScene,1)
            if y <= round(size(locationOfScene,1)*TrainPercentage/100)
                trainingDB(trainingIndex) = locationOfScene(y);
                trainingIndex = trainingIndex + 1;
            elseif (round(size(locationOfScene,1)*TrainPercentage/100) < y) && (y <= round(size(locationOfScene,1)*(TrainPercentage+TestPercentage)/100))
                testDB(testIndex) = locationOfScene(y);
                testIndex = testIndex + 1;
            else
                validationDB(ValIndex) = locationOfScene(y);
                ValIndex = ValIndex + 1;
            end
        end
        if size(locationOfScene,1)<=ClassTreshold
            fprintf('%s, ',thisScene{:});
        end  
end
fprintf('\n');

% Define the sizes of the DB
trainingDBSize = size(trainingDB,2);
testDBSize = size(testDB,2);
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
        
        if rem(index,100)==0
            fprintf('%d ~ %d of %d \n',index-99,index,trainingDBSize);
        end
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
    %------------------------------Train & Validate SVM------------------------
    disp('Train And Validate SVM')
    %New database of validate images 
    %for index = 1:trainingDBSize
        sceneTrainTypes(:) = sceneTypes(trainingDB(:));
    %end
    
    %Prepare ValidationDB
    for index = 1:validationDBSize
        %Normalize ValidationDB
        im_temp = single(images(:,:,:,validationDB(index))) ; % note: 0-255 range
        im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
        im_(:,:,:,index) = im_temp - net.normalization.averageImage ;
        %Run CNN on ValidationDB
        resVal = vl_simplenn(net, im_(:,:,:,index)) ; 
        lastFCVal = squeeze(gather(resVal(lastFClayer+1).x));
    end
    
    for c = C %Better ==> Stopping condition?
        %fprintf('C= %f\n',c);
        clear startNegatives
        startNegatives = zeros(amountOfScenes,trainingDBSize);
        %------------------------------Train SVM-----------------------------------
        for i = 1:amountOfScenes
            thisScene = uniqueScenes(i);
            %disp(thisScene)
            positives = find(strcmp(sceneTrainTypes,thisScene));
            negatives = find(strcmp(sceneTrainTypes,thisScene)==0);
            startNegatives(i,1:min(3*size(positives,2),size(negatives,2))) = randsample(negatives,min(3*size(positives,2),size(negatives,2)));
            startNegativesTemp1 = startNegatives(i,:);
            startNegativesTemp = startNegativesTemp1(startNegativesTemp1~=0);
            SVMLabel = ones(size(startNegativesTemp,2)+size(positives,2),1);
            SVMLabel(1:size(startNegativesTemp,2)) = -1;
            lastFCOfTraining = lastFC(:,[startNegativesTemp,positives]);
            
            [X,Y,INFO] = vl_svmtrain(lastFCOfTraining,SVMLabel,c);
            if i==1
                WTemp = X;
                BTemp = Y;
            else
                WTemp = [WTemp,X];
                BTemp = [BTemp,Y];
            end
        end
        %--------------------------Cross-Validate SVM----------------------------------
        correct = 0;
        for index = 1:validationDBSize
            for i = 1:amountOfScenes
                scoresVal(:,i) = WTemp(:,i)'*lastFCVal + BTemp(i) ;
            end
            [bestScore(index), best(index)] = max(scoresVal) ;
            if strcmp(uniqueScenes{best(index)}, sceneTypes(validationDB(index)))
                correct = correct + 1;
            end
        end
        performance(find(C==c)) = correct/validationDBSize;
    end
    
%     figure;
%     scatter(C,performance);
    
    
    clear WTemp BTemp c correct best bestScore
    %--------------------------SVM with best param----------------------------------
    [~,CBest] = max(performance);
    for i = 1:amountOfScenes
            thisScene = uniqueScenes(i);
            %disp(thisScene)
            positives = find(strcmp(sceneTrainTypes,thisScene));
            negatives = find(strcmp(sceneTrainTypes,thisScene)==0);
            startNegativesTemp1 = startNegatives(i,:);
            startNegativesTemp = startNegativesTemp1(startNegativesTemp1~=0);
            SVMLabel = ones(size(startNegativesTemp,2)+size(positives,2),1);
            SVMLabel(1:size(startNegativesTemp,2)) = -1;
            lastFCOfTraining = lastFC(:,[startNegativesTemp,positives]);
            
            [X,Y,INFO] = vl_svmtrain(lastFCOfTraining,SVMLabel,C(CBest));
            if i==1
                W = X;
                B = Y;
            else
                W = [W,X];
                B = [B,Y];
            end
    end
    %---------------------------------------------------------------------------------------------------
    %---------------------------------------------------------------------------------------------------
    %--------------------------Hard Negative Mining----------------------------------
    clear performance
    for c = C %Better ==> Stopping condition?
        fprintf('C= %f\n',c);
        %------------------------------Retrain SVM-----------------------------------
        for i = 1:amountOfScenes
            clear best bestScore positives negatives startNegativesTemp
            thisScene = uniqueScenes(i);
            %disp(thisScene)
            positives = find(strcmp(sceneTrainTypes,thisScene));
            negatives = find(strcmp(sceneTrainTypes,thisScene)==0);
            
            startNegativesTemp1 = startNegatives(i,:);
            startNegativesTemp = startNegativesTemp1(startNegativesTemp1~=0);
            for index = 1:size(startNegativesTemp,2)
                negatives(negatives==startNegativesTemp(index)) = []; %Get the negatives out that are already done
            end
            for index = 1:size(negatives,2) %All unused negatives
                for index2 = 1:amountOfScenes
                    scoresHardMining(:,index2) = W(:,index2)'*lastFC(:,negatives(index)) + B(index2) ;
                end
                [bestScore(index), best(index)] = max(scoresHardMining) ;
            end
            
            if exist('best') == 1
                newNegatives = negatives(best==i); %FP
            else
                newNegatives = [];
            end
            %Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),3) = Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),3) + 1; %#FP
            clear negatives lastFCOfTraining SVMLabel
            negatives = [startNegativesTemp, newNegatives];
            SVMLabel = ones(size(negatives,2)+size(positives,2),1);
            SVMLabel(1:size(negatives,2)) = -1;
            lastFCOfTraining = lastFC(:,[negatives,positives]);
            
            [X,Y,INFO] = vl_svmtrain(lastFCOfTraining,SVMLabel,c);
            if i==1
                WTemp = X;
                BTemp = Y;
            else
                WTemp = [WTemp,X];
                BTemp = [BTemp,Y];
            end
        end
        %--------------------------Cross-Validate SVM----------------------------------
        correct = 0;
        for index = 1:validationDBSize
            for i = 1:amountOfScenes
                scoresVal(:,i) = WTemp(:,i)'*lastFCVal + BTemp(i) ;
            end
            [bestScore(index), best(index)] = max(scoresVal) ;
            if strcmp(uniqueScenes{best(index)}, sceneTypes(validationDB(index)))
                correct = correct + 1;
            end
        end
        performance(find(C==c)) = correct/validationDBSize;
    end
    
%     figure;
%     scatter(C,performance);
    
    clear c correct best bestScore
    %--------------------------SVM with best param----------------------------------
    [~,CBest] = max(performance);
    for i = 1:amountOfScenes
%             thisScene = uniqueScenes(i);
%             %disp(thisScene)
%             positives = find(strcmp(sceneTrainTypes,thisScene));
%             negatives = find(strcmp(sceneTrainTypes,thisScene)==0);
%             startNegatives = randsample(negatives,min(3*size(positives,2),size(negatives,2)));
%             SVMLabel = ones(size(startNegatives,2)+size(positives,2),1);
%             SVMLabel(1:size(startNegatives,2)) = -1;
%             lastFCOfTraining = lastFC(:,[startNegatives,positives]);
%             
%             [X,Y,INFO] = vl_svmtrain(lastFCOfTraining,SVMLabel,C(CBest));
%             if i==1
%                 W = X;
%                 B = Y;
%             else
%                 W = [W,X];
%                 B = [B,Y];
%             end
            
            
            
            
            
            clear best bestScore positives negatives startNegativesTemp
            thisScene = uniqueScenes(i);
            %disp(thisScene)
            positives = find(strcmp(sceneTrainTypes,thisScene));
            negatives = find(strcmp(sceneTrainTypes,thisScene)==0);
            
            startNegativesTemp1 = startNegatives(i,:);
            startNegativesTemp = startNegativesTemp1(startNegativesTemp1~=0);
            for index = 1:size(startNegativesTemp,2)
                negatives(negatives==startNegativesTemp(index)) = []; %Get the negatives out that are already done
            end
            for index = 1:size(negatives,2) %All unused negatives
                for index2 = 1:amountOfScenes
                    scoresHardMining(:,index2) = WTemp(:,index2)'*lastFC(:,negatives(index)) + BTemp(index2) ;
                end
                [bestScore(index), best(index)] = max(scoresHardMining) ;
            end
            
            if exist('best') == 1
                newNegatives = negatives(best==i); %FP
            else
                newNegatives = [];
            end
            %Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),3) = Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),3) + 1; %#FP
            clear negatives lastFCOfTraining SVMLabel
            negatives = [startNegativesTemp, newNegatives];
            SVMLabel = ones(size(negatives,2)+size(positives,2),1);
            SVMLabel(1:size(negatives,2)) = -1;
            lastFCOfTraining = lastFC(:,[negatives,positives]);
            
            [X,Y,INFO] = vl_svmtrain(lastFCOfTraining,SVMLabel,C(CBest));
            if i==1
                W = X;
                B = Y;
            else
                W = [W,X];
                B = [B,Y];
            end
    end
    
    
    %---------------------------------------------------------------------------------------------------
    %---------------------------------------------------------------------------------------------------
    
    save('svm.mat','W','B','uniqueScenes');
    disp('Training finished')
    %--------------------------------------------------------------------------
else
    load('svm.mat');
end

%scatter(C,performance);


%---------------------------Test (ROC)---------------------------------------
disp('Start tests')
correct = 0;
Result = zeros(amountOfScenes,3); %columns: 1. # scenes in testDB. 2. # True Positives. 3. #False Positives
for index = 1:testDBSize
    %Normalize
    im_temp = single(images(:,:,:,testDB(index))) ; % note: 0-255 range
    im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
    im_(:,:,:,index) = im_temp - net.normalization.averageImage ;
    %Run CNN
    resTest = vl_simplenn(net, im_(:,:,:,index)) ; 
    lastFCTest = squeeze(gather(resTest(lastFClayer+1).x));
    
    for i = 1:amountOfScenes
        scoresTest(:,i) = W(:,i)'*lastFCTest + B(i) ;
    end
    
    [bestScore(index), best(index)] = max(scoresTest) ;
    Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),1) = Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),1) + 1; %#in testDB
    
    
    %Check against given scene
    if strcmp(uniqueScenes{best(index)}, sceneTypes(testDB(index)))
        Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),2) = Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),2) + 1; %#TP
        %fprintf('CORRECT: %s \n',uniqueScenes{best(index)});
    else
        Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),3) = Result(find(strcmp(sceneTypes(testDB(index)),uniqueScenes)),3) + 1; %#FP
        %fprintf('Wrong: %s but correct is %s \n',uniqueScenes{best(index)},sceneTypes{testDB(index)});
    end
    
    if rem(index,100)==0
            fprintf('%d ~ %d of %d \n',index-99,index,testDBSize);
    end
end
sumResult = sum(Result);
fprintf('Out of a testDB of %d ==> %dTP and %dFP \n',sumResult(1),sumResult(2),sumResult(3));
disp('Tests finished')
%fprintf('Result: %d out of %d are correct\n',correct,testDBSize);
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