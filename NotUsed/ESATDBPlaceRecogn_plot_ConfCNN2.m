clear all
close all
clc

TrainingCoordinates = makeTrainingCoordinates();
addpath data deps/matconvnet-1.0-beta16 data/ESAT-DB

%Run setup before! to compile matconvnet
%%
%------------------------VARIABLES-----------------------------------------
PlotOn = 0; %Plot Debugging figures

%WARNING: If change of testDB ==> RunCNN, RunConf, calcScenesTestDB, RunConfScene =1
testDB = 4; %Select the testDB: 1 (same day) or 2 (after ~2 months)

lastFClayer = 13;

%WARNING: If change of edgeThreshold ==> Put RunConfScene AND RunConfCNN AND RunConfObjects to 1
edgeThresholdTraining = 0;%0.05;
edgeThresholdTest = 0.075;%0.05;

RunCNN = 0;     %1 = run the CNN, 0 = Load the CNN
RunConfCNN = 0;    %1 = recalc the Conf. matrix, 0 = Load the Conf. Matrix
PlotRoute = 0;  %1 = plot the route on a floorplan

%Scene Recognition
calcScenesTrainingDB = 0;   %1 if recalc of the scenes for the trainingDB is necessary.
calcScenesTestDB = 0;       %1 if recalc of the scenes for the testDB is necessary.
RunConfScene = 0;           %1 = recalc the Conf. matrix for the Scene Recognition, 0 = Load the Conf. Matrix

%Object Localisation
calcObjLocTraining = 0;
calcObjLocTest = 0;
n_box_max = 5; % Max amount of boxes to be used for object recognition

%Object Recognition
calcObjRecTraining = 0;
calcObjRecTest = 0;
RunConfObjects = 0;
n_labels_max = 5; %Max amount of recognized objects per box

ConfMatCNN = 0.600; % Multiplied with the CNN feature CNN, and 1-ConfMatCNN is multiplied with the Scene Recogn Conf Matrix.
ConfMatObj = 0.100; % 
ConfMatScene = 0.300;
if ConfMatCNN+ConfMatObj+ConfMatScene ~=1
    error('Check the Confusion Matrix parameters.');
end

%Spatial Continuity check
d = 35; % Length of evaluation window
epsilon = 37;% maximal jumps of trainingframes in this evaluation window

%Particle Filter
%FeatureDetectNoiseStDev = 200;  %Standard deviation on calculated difference of features
SpeedStDev = 2;                 %Standard deviation on calculated speed
Speed = 1;                      %speed of walking
RandPercentage = 0.01;           %Percentage of the particles to be randomized (1 = 100%)
N = 2500;                       %Amount of particles
PlotPF = 0;                     %1 = plot the PF for debugging & testing

locationMode = 2; %1 = No correction, 2 = Spatial Continuity, 3 = Particle Filtering

%Error calculation
widthRoom68 = 3; %used to calculate the error
RunError = 1; %1 = recalc the error for every method; 0 = load this error
%--------------------------------------------------------------------------
%%
% load the pre-trained CNN
net = load('data/cnns/imagenet-matconvnet-vgg-m.mat') ;
if sum(size(net.meta.normalization.averageImage)) == 4
    averageImage(:,:,1) = net.meta.normalization.averageImage(1) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,2) = net.meta.normalization.averageImage(2) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,3) = net.meta.normalization.averageImage(3) * ones(net.meta.normalization.imageSize(1:2));
else
    averageImage = net.meta.normalization.averageImage;
end

disp('loading ESAT DB')
switch testDB
    case 1
        T = load('data/ESAT-DB/mat/test.mat');
        TestCoordinates = makeTestCoordinates();
        datapath = 'data/test/1/';
    case 2
        T = load('data/ESAT-DB/mat/test2.mat');
        TestCoordinates = makeTest2Coordinates();
        datapath = 'data/test/2/';
    case 3
        T = load('data/ESAT-DB/mat/test3.mat');
        TestCoordinates = makeTest3Coordinates();
        datapath = 'data/test/3/';
    case 4
        T = load('data/ESAT-DB/mat/test4.mat');
        TestCoordinates = makeTest4Coordinates();
        datapath = 'data/test/4/';
    case 5
        T = load('data/ESAT-DB/mat/test5.mat');
        TestCoordinates = makeTest5Coordinates();
        datapath = 'data/test/5/';
    case 6
        T = load('data/ESAT-DB/mat/test6.mat');
        TestCoordinates = makeTest6Coordinates();
        datapath = 'data/test/6/';
end
testImg_original = T.img;
clear T
T = load('data/ESAT-DB/mat/training.mat');
trainingImg_original = T.img;
clear T
disp('DB loaded')

% Define the sizes of the DB
trainingDBSize_original = size(trainingImg_original,4);
testDBSize_original = size(testImg_original,4);

%Setup MatConvNet
delete(gcp('nocreate'))
run deps/matconvnet-1.0-beta16/matlab/vl_setupnn;

%%
%--------------------------Edge  Detection---------------------------------
%Leave the images out of the training & test if the amount of edges is below a
%specific treshold
%Parfor gives speedimprovement for trainingDB: 17sec -> 10sec

trainingImg = trainingImg_original;
testImg = testImg_original;

%uselessTrainingImg = zeros(trainingDBSize,1);
TrainingToDelete = [];
parfor index = 1:trainingDBSize_original
    [~,threshOut_training(index)] = edge(rgb2gray(trainingImg(:,:,:,index)));
    if threshOut_training(index) < edgeThresholdTraining
        TrainingToDelete = [TrainingToDelete,index];
        %uselessTrainingImg(index) = 1;
    end
end
trainingImg(:,:,:,TrainingToDelete(:)) = [];
TrainingCoordinates(TrainingToDelete(:),:) = [];

%uselessTestImg = zeros(testDBSize,1);
TestToDelete = [];
parfor index = 1:testDBSize_original
    [~,threshOut_test(index)] = edge(rgb2gray(testImg(:,:,:,index)));
    if threshOut_test(index) < edgeThresholdTest
        TestToDelete = [TestToDelete,index];
        %uselessTestImg(index) = 1;
    end
end
% testImg(:,:,:,TestToDelete(:)) = [];
% TestCoordinates(TestToDelete(:),:) = [];

% Define the sizes of the new DB
trainingDBSize = size(trainingImg,4);
testDBSize = size(testImg,4);
%--------------------------------------------------------------------------

%%
if RunCNN




    % ------------load and preprocess the images---------------------------------
    disp('Normalization')
    imSize = net.meta.normalization.imageSize(1:2);
    for index = 1:trainingDBSize_original
        im_temp = single(trainingImg_original(:,:,:,index)) ; % note: 0-255 range
        im_temp = imresize(im_temp, imSize) ;
        trainingImgNorm(:,:,:,index) = im_temp - averageImage ;
    end
    clear im_temp index
    for index = 1:testDBSize_original
        im_temp = single(testImg_original(:,:,:,index)) ; % note: 0-255 range
        im_temp = imresize(im_temp, imSize) ;
        testImgNorm(:,:,:,index) = im_temp - averageImage ;
    end
    clear im_temp index imSize
    disp('Normalization finished')
    %--------------------------------------------------------------------------

    
    % ---------------------------------Run CNN---------------------------------
    disp('Run CNN')
    delete(gcp('nocreate'))
    for index = 1:trainingDBSize_original
        res = vl_simplenn(net, trainingImgNorm(:,:,:,index)) ;
        
        lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
        lastFCtraining(:,index) = lastFCtemp(:);
        
        if rem(index,100)==0
            fprintf('training %d ~ %d of %d \n',index-99,index,trainingDBSize_original);
        end
    end
    if exist([datapath,'lastFCesatDB.mat'], 'file')
        save([datapath,'lastFCesatDB.mat'],'lastFCtraining','-append');
    else
        save([datapath,'lastFCesatDB.mat'],'lastFCtraining');
    end
    clear lastFCtemp index res
    
    %Same for testDB
    for index = 1:testDBSize_original
        res = vl_simplenn(net, testImgNorm(:,:,:,index)) ;
        
        lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
        lastFCtest(:,index) = lastFCtemp(:);
        
        if rem(index,100)==0
            fprintf('test %d ~ %d of %d \n',index-99,index,testDBSize_original);
        end
    end
    if exist([datapath,'lastFCesatDB.mat'], 'file')
        save([datapath,'lastFCesatDB.mat'],'lastFCtest','-append');
    else
        save([datapath,'lastFCesatDB.mat'],'lastFCtest');
    end
    clear lastFCtemp res
    disp('CNN finished')
    %--------------------------------------------------------------------------
else
    disp('CNN not recalculated')
    load([datapath,'lastFCesatDB.mat']);
end

lastFCtraining_original = lastFCtraining;
lastFCtest_original = lastFCtest;

lastFCtraining(:,TrainingToDelete(:)) = [];
% lastFCtest(:,TestToDelete(:)) = [];

%%
%-------------------------------Scene Recognition--------------------------
%GOAL: Save for every test image the scores for every scenetype. (Thus
%a score for every scene in the testDB) This will be added to the
%ConfusionMatrix and used for the localisation.

disp('Load scenes & SVM')
load('data/scene/ESATsvm.mat');
disp('Scenes & SVM loaded')

%Retrain or Load the TrainingDB
if calcScenesTrainingDB
    disp('Recalculate Scenes for the trainingDB')
    
    for index = 1:trainingDBSize_original
        for i = 1:size(uniqueScenes,1)
            scoresTraining(index,i) = W(:,i)'*lastFCtraining_original(:,index) + B(i) ;
        end
    end
    
    if exist([datapath,'ScenesEsatDB.mat'], 'file')
        save([datapath,'ScenesEsatDB.mat'],'scoresTraining','-append');
    else
        save([datapath,'ScenesEsatDB.mat'],'scoresTraining');
    end
    disp('Scenes saved for the trainingDB')
else
    disp('Scenes for the TrainingDB not recalculated')
    load([datapath,'ScenesEsatDB.mat'],'scoresTraining');
end
scoresTraining(TrainingToDelete(:),:) = [];

%Retrain or Load the TestDB
if calcScenesTestDB
    disp('recalculate scenes for the testDB')
    
    for index = 1:testDBSize_original
        for i = 1:size(uniqueScenes,1)
            scoresTest(index,i) = W(:,i)'*lastFCtest_original(:,index) + B(i) ;
        end
        
        %Save the best scene with the score
        [bestScoreScene(index), bestScene(index)] = max(scoresTest(index,:)) ;
    end
    
    if exist([datapath,'ScenesEsatDB.mat'], 'file')
        save([datapath,'ScenesEsatDB.mat'],'scoresTest','bestScoreScene','bestScene','-append');
    else
        save([datapath,'ScenesEsatDB.mat'],'scoresTest','bestScoreScene','bestScene');
    end
    disp('Scenes saved for the testDB')
else 
    disp('Scenes for the testDB not recalculated')
    load([datapath,'ScenesEsatDB.mat'],'scoresTest','bestScoreScene','bestScene');
end
% scoresTest(TestToDelete(:),:) = [];
% bestScoreScene(TestToDelete(:)) = [];
% bestScene(TestToDelete(:)) = [];

%Make a temporary confusionMatrix for the scene-recognition
if RunConfScene
    disp('Start to compare scenes')
    confusionMatrixSceneRecogn = zeros(trainingDBSize);
    for index = 1:testDBSize
        for i = 1:trainingDBSize
            confusionMatrixSceneRecogn(i,index) = norm(scoresTest(index,:)-scoresTraining(i,:));
        end
%         if rem(index,100)==0
%                 fprintf('Scene Calc. %d ~ %d of %d \n',index-99,index,testDBSize);
%         end
    end
    if exist([datapath,'confMatrix.mat'], 'file')
        save([datapath,'confMatrix.mat'],'confusionMatrixSceneRecogn','-append');
    else
        save([datapath,'confMatrix.mat'],'confusionMatrixSceneRecogn');
    end
    disp('ConfusionMatrix of Scenes saved')
else
    disp('ConfusionMatrix of Scenes not recalculated')
    load([datapath,'confMatrix.mat'],'confusionMatrixSceneRecogn');
end
confusionMatrixSceneRecogn = (confusionMatrixSceneRecogn - min(min(confusionMatrixSceneRecogn)))./max(max(confusionMatrixSceneRecogn));

if PlotOn
    figure;
    imagesc(confusionMatrixSceneRecogn)
    title('Confusion Matrix Scene Recognition')
    ylabel('Training Image')
    xlabel('Test Image')
end
%--------------------------------------------------------------------------

%%
%-------------------Object Localisation & Recognition----------------------
%GOAL: Save for every image the scores for every objectcategory and the objectlocation on the image. (Thus
%a score for every object in the DB) This will be added to the
%ConfusionMatrix and used for the localisation.

%LOCALISATION using DeepProposals from 'A. Gohdrati et Al'
if calcObjLocTraining
    disp('Recalculate object locations for the trainingDB')
    delete(gcp('nocreate'))
    trainingObjectLocation = calc_object_locations( n_box_max, trainingImg_original );
    
    if exist([datapath,'Objects.mat'], 'file')
        save([datapath,'Objects.mat'],'trainingObjectLocation','-append');
    else
        save([datapath,'Objects.mat'],'trainingObjectLocation');
    end
    disp('Object locations saved for the trainingDB')
else
    disp('Object locations for the trainingDB not recalculated')
    load([datapath,'Objects.mat'],'trainingObjectLocation');
end

if calcObjLocTest
    disp('Recalculate object locations for the testDB')
    delete(gcp('nocreate'))
    testObjectLocation = calc_object_locations( n_box_max, testImg_original );
    
    if exist([datapath,'Objects.mat'], 'file')
        save([datapath,'Objects.mat'],'testObjectLocation','-append');
    else
        save([datapath,'Objects.mat'],'testObjectLocation');
    end
    disp('Object locations saved for the testDB')
else
    disp('Object locations for the testDB not recalculated')
    load([datapath,'Objects.mat'],'testObjectLocation');
end


%RECOGNITION
if calcObjRecTraining
    disp('Recalculate object recognition for the trainingDB')
    delete(gcp('nocreate'))
    
    trainingObjectRecognition = calc_object_recognition( trainingImg_original, trainingObjectLocation, net, n_labels_max );
    
    disp('Objects recognized, now selecting one object for each box')
    
    % the classification result
    bestScoreObject_training = squeeze(trainingObjectRecognition(1,2,:,:)) ; % bestScoreObject_training(Frame_number,img_Number)
    objectNumber_training = squeeze(trainingObjectRecognition(1,1,:,:)) ; % objectNumber_training(Frame_number,img_Number)
    objectName_training = reshape(net.meta.classes.description(objectNumber_training(:)),size(objectNumber_training));
   
    if exist([datapath,'Objects.mat'], 'file')
        save([datapath,'Objects.mat'],'trainingObjectRecognition','bestScoreObject_training','objectNumber_training','objectName_training','-append');
    else
        save([datapath,'Objects.mat'],'trainingObjectRecognition','bestScoreObject_training','objectNumber_training','objectName_training');
    end
    disp('Object recognition saved for the trainingDB')
else
    disp('Object recognition for the trainingDB not recalculated')
    load([datapath,'Objects.mat'],'trainingObjectRecognition','bestScoreObject_training','objectNumber_training','objectName_training');
end

if calcObjRecTest
    disp('Recalculate object recognition for the testDB')
    delete(gcp('nocreate'))
    
    testObjectRecognition = calc_object_recognition( testImg_original, testObjectLocation, net, n_labels_max );
    
    disp('Objects recognized, now selecting one object for each box')
    
    % the classification result
    bestScoreObject_test = squeeze(testObjectRecognition(1,2,:,:)) ; % bestScoreObject_test(Frame_number,img_Number)
    objectNumber_test = squeeze(testObjectRecognition(1,1,:,:)) ; % objectNumber_test(Frame_number,img_Number)
    objectName_test = reshape(net.meta.classes.description(objectNumber_test(:)),size(objectNumber_test));
    
    if exist([datapath,'Objects.mat'], 'file')
        save([datapath,'Objects.mat'],'testObjectRecognition','bestScoreObject_test','objectNumber_test','objectName_test','-append');
    else
        save([datapath,'Objects.mat'],'testObjectRecognition','bestScoreObject_test','objectNumber_test','objectName_test');
    end
    disp('Object recognition saved for the testDB')
else
    disp('Object recognition for the testDB not recalculated')
    load([datapath,'Objects.mat'],'testObjectRecognition','bestScoreObject_test','objectNumber_test','objectName_test');
end


trainingObjectLocation(:,:,TrainingToDelete(:)) = [];
% testObjectLocation(:,:,TestToDelete(:)) = [];
% 
% testObjectRecognition(:,:,:,TestToDelete(:)) = [];
% bestScoreObject_test(:,TestToDelete(:)) = [];
% objectName_test(:,TestToDelete(:)) = [];
% objectNumber_test(:,TestToDelete(:)) = [];

trainingObjectRecognition(:,:,:,TrainingToDelete(:)) = [];
bestScoreObject_training(:,TrainingToDelete(:)) = [];
objectName_training(:,TrainingToDelete(:)) = [];
% objectNumber_training(:,TestToDelete(:)) = [];
%--------------------------------------------------------------------------

%%
%------------------Confusion Matrix Object recognition---------------------
if RunConfObjects
    disp('Start calculating the confusion matrix for the Object Recognition')
    confusionMatrixObjects = ones(trainingDBSize).*n_box_max;
    for index = 1:testDBSize
        for i = 1:trainingDBSize
            test = objectNumber_test(:,index);
            
            for q = 1:n_box_max
                for z = 1:size(test,1)
                    if objectNumber_training(q,i) == test(z)
                        confusionMatrixObjects(i,index) = confusionMatrixObjects(i,index) - 1;
                        test(z) = [];
                        break
                    end
                end
            end
            
        end
        
        
%         if rem(index,100)==0
%              fprintf('Confusion Calc. %d ~ %d of %d \n',index-99,index,testDBSize);
%         end    
    end

    if exist([datapath,'confMatrix.mat'], 'file')
        save([datapath,'confMatrix.mat'],'confusionMatrixObjects','-append');
    else
        save([datapath,'confMatrix.mat'],'confusionMatrixObjects');
    end
else
    disp('ConfusionMatrix for the Object Recognition not recalculated')
    load([datapath,'confMatrix.mat'],'confusionMatrixObjects');
end
confusionMatrixObjects = (confusionMatrixObjects - min(min(confusionMatrixObjects)))./max(max(confusionMatrixObjects));

if PlotOn
    figure;
    imagesc(confusionMatrixObjects)
    title('Confusion Matrix Object recognition')
    ylabel('Training Image')
    xlabel('Test Image')
end
%--------------------------------------------------------------------------

%%
%------------------------Confusion Matrix CNN Features---------------------
if RunConfCNN
    disp('Start calculating the confusion matrix for the CNN features')
    confusionMatrixCNNFeat = zeros(trainingDBSize);
    for index = 1:testDBSize
        for i = 1:trainingDBSize
            confusionMatrixCNNFeat(i,index) = norm(lastFCtest(:,index)-lastFCtraining(:,i));
        end
%         if rem(index,100)==0
%                 fprintf('Confusion Calc. %d ~ %d of %d \n',index-99,index,testDBSize);
%         end
    end
    if exist([datapath,'confMatrix.mat'], 'file')
        save([datapath,'confMatrix.mat'],'confusionMatrixCNNFeat','-append');
    else
        save([datapath,'confMatrix.mat'],'confusionMatrixCNNFeat');
    end
else
    disp('ConfusionMatrix CNN features not recalculated')
    load([datapath,'confMatrix.mat'],'confusionMatrixCNNFeat');
end
confusionMatrixCNNFeat = (confusionMatrixCNNFeat - min(min(confusionMatrixCNNFeat)))./max(max(confusionMatrixCNNFeat));

if PlotOn
    figure;
    imagesc(confusionMatrixCNNFeat)
    title('Confusion Matrix CNN features')
    ylabel('Training Image')
    xlabel('Test Image')
end
%--------------------------------------------------------------------------
%%
if RunError
    plotindex = 1;
    for ConfMatCNN = [0,0.6,1] % Multiplied with the CNN feature CNN, and 1-ConfMatCNN is multiplied with the Scene Recogn Conf Matrix.
        if ConfMatCNN == 0
            abcd = [0,1];
        elseif ConfMatCNN == 0.6
            abcd = 0.1;
        else
            abcd = 0;
        end
        for ConfMatObj = abcd
            ConfMatScene = 1-ConfMatCNN-ConfMatObj;

            %%
            %------------------------Combine Confusion Matrices------------------------
            disp('Start combining the confusion matrices')

            %confusionMatrix = ConfMatCNN .* confusionMatrixCNNFeat + (1-ConfMatCNN) .* confusionMatrixSceneRecogn;
            %confusionMatrix = confusionMatrixCNNFeat .* confusionMatrixSceneRecogn;
            confusionMatrix = ConfMatCNN .* confusionMatrixCNNFeat + ConfMatScene .* confusionMatrixSceneRecogn + ConfMatObj .* confusionMatrixObjects;
            if PlotOn
                figure;
                imagesc(confusionMatrix)
                title(['Combined Confusion Matrix with ConfMatCNN=',num2str(ConfMatCNN),'; ConfMatScene=',num2str(ConfMatScene),'; ConfMatObj=',num2str(ConfMatObj)])
                ylabel('Training Image')
                xlabel('Test Image')
            end
            %--------------------------------------------------------------------------

            %%
            %------------------------Select Lowest difference--------------------------
            disp('Search lowest difference')
            parfor index = 1:testDBSize
                [ResultValue(index),Result(index)] = min(confusionMatrix(:,index));
            end
            if PlotOn
                figure;
                plot(Result,'g')
                hold on
            end
            %%
            %------------------------Spatial Continuity check--------------------------
            for index = d:testDBSize
                P(index) = 1;
                for u = index-d+2:index
                    if abs(Result(u-1)-Result(u)) > epsilon
                        P(index) = 0;
                        break;
                    end
                end
            end

            %Only motion model if P=0
            ResultSC(1) = Result(1);
            for index = 2:testDBSize
                if P(index) == 1
                    ResultSC(index) = Result(index);
                else
                    ResultSC(index) = ResultSC(index-1)+1;
                end
            end
            ResultSC(ResultSC>trainingDBSize) = trainingDBSize;
            if PlotOn
                plot(ResultSC,'r')
                %plot(Resultnew-Result,'g')
                hold off
                title(['Green = initial, Red = after Spatial Continuity Check with: epsilon = ' num2str(epsilon) '; d = ' num2str(d)])
                xlabel('test images')
                ylabel('training images')
            end

            %%
            %-------------------------------Sequential Filter--------------------------
            % clear u
            % u = 1;
            % d=50;
            % for index = d:testDBSize
            %     X = ones(d,2);
            %     X(:,2) = index-d+1:index;
            %     Y = reshape(Resultnew(index-d+1:index),[d,1]);
            %     BetaAlpha(:,u) = X\Y; 
            %     u = u+1;
            % end





            %%
            %To delete/fix
            %FeatureDetectNoiseStDev = 0.0775;%max(range(confusionMatrix))/2;






            %%
            %-------------------------------Particle Filter--------------------------
            %Initialize particles
            particles = round(rand(N,1)*(trainingDBSize-1)+1);

            if PlotPF
                figure('units','normalized','outerposition',[0 0 1 1]);
                vPF = VideoWriter([datapath,'VideoPF.avi']);
                open(vPF)
            end
            w=ones(N,1)./N;
            ResultPF = zeros(1,testDBSize);
            for index = 1:testDBSize

                %Check this!
                FeatureDetectNoiseStDev = sqrt(var(confusionMatrix(:,index)));

                if ~any(index == TestToDelete)
                    %Create weights using the normal distribution pdf & normalize
                    w=ones(N,1)./N;
                    w = w.*(1/(sqrt(2*pi)*FeatureDetectNoiseStDev)*exp(-(  confusionMatrix(particles(:),index)).^2/(2*FeatureDetectNoiseStDev^2)));
                    w = w/sum(w);
                end

                if PlotPF
                    %plot the particles
                    subplot(3,3,1:3);
                    stem(particles(:),w*10000);
                    axis([0 trainingDBSize 0 inf])
                    title('Particle weights');
                    xlabel('Training Image');
                    ylabel('Weight (*10000)');
                end

                if ~any(index == TestToDelete)
                    %Resample the particles = leave out the unlikely particles
                    u = rand(N,1);
                    wc = cumsum(w);
                    [~,ind1] = sort([u;wc]);
                    ind=find(ind1<=N)-(0:N-1)';
                    particles=particles(ind);
                end


                %motion model
                particles = round(particles + Speed + SpeedStDev*randn(size(particles)));
                particles(particles<=0)=1;
                particles(particles>=trainingDBSize)=trainingDBSize;

                if ~any(index == TestToDelete)
                    %Randomize a specific percentage to avoid locked particles
                    particles(round(rand(ceil(N*RandPercentage),1)*(N-1)+1)) = round(rand(ceil(N*RandPercentage),1)*(trainingDBSize-1)+1);
                end
                %Keep result of the PF
                ResultPF(index) = mode(particles);

                if PlotPF
                    subplot(3,3,4);
                    imshow(testImg(:,:,:,index));
                    if ~any(index == TestToDelete)
                        title(['Test Image ',num2str(index)]);
                    else
                        title(['Test Image ',num2str(index),' ignored']);
                    end

                    subplot(3,3,5);
                    imshow(trainingImg(:,:,:,ResultPF(index)));
                    title(['Training Image ',num2str(ResultPF(index))]);

                    subplot(3,3,7:9);
                    histogram(particles,round(trainingDBSize/10));
                    hold on
                    axis([0 trainingDBSize 0 inf])
                    title('Particle histogram');
                    xlabel('Training Image');
                    ylabel('Amount of particles');


                    line([ResultPF(index) ResultPF(index)], [0 1000], 'color','r');
                    hold off

                    subplot(3,3,6);
                    if threshOut_test(index) < edgeThresholdTest
                        bar([threshOut_test(index) threshOut_training(ResultPF(index))],'r');
                    else
                        bar([threshOut_test(index) threshOut_training(ResultPF(index))]);
                    end
                    axis([0.5 2.5 0 0.2])
                    title('Edges detected. first = testDB, last = trainingDB-result from PF');
                    xlabel('Test - Training');
                    ylabel('edges');


                    frame = getframe(gcf);
                    writeVideo(vPF,frame)
                end
                storedParticles(:,index) = particles;
            end
            if PlotPF
                close(vPF);
            end
            if PlotOn
                figure;
                plot(Result,'g')
                hold on
                plot(ResultPF,'r')
                hold off
                title({['Green = initial, Red = after Particle Filtering with N=',num2str(N),'; Speed=',num2str(Speed)],['RandPercentage=',num2str(RandPercentage),'; SpeedStDev=',num2str(SpeedStDev),'; FeatureDetectNoiseStDev=',num2str(FeatureDetectNoiseStDev)]});
                xlabel('Test Image');
                ylabel('Training Image');
            end
            %--------------------------------------------------------------------------

            %%
            %-----------------------------Calculate the error--------------------------
            for i = 1:3
                %Find the calculated test image coordinates
                switch i %locationMode
                    case 1
                        % No post-processing
                        testLocations = [TrainingCoordinates(Result(1,:),1),TrainingCoordinates(Result(1,:),2)];
                        description = 'no post-processing';
                    case 2
                        % Spatial Continuity filter
                        testLocations = [TrainingCoordinates(ResultSC(1,:),1),TrainingCoordinates(ResultSC(1,:),2)];
                        description = ['the spatial continuity filter with: epsilon=' num2str(epsilon) '; d=' num2str(d)];
                    case 3
                        % Particle Filter
                        testLocations = [TrainingCoordinates(ResultPF(1,:),1),TrainingCoordinates(ResultPF(1,:),2)];
                        description = ['the particle filter with N=',num2str(N),'; Speed=',num2str(Speed),'; RandPercentage=',num2str(RandPercentage),'; SpeedStDev=',num2str(SpeedStDev)];
                end

                % Calc the amount of meter per pixel on the floorplan
                MeterPixel = widthRoom68/(506-480);

                %testCoordinates = Groundtruth
                error = TestCoordinates - testLocations;
                errorDistance = sqrt(error(:,1).^2 + error(:,2).^2).*MeterPixel;
                if PlotOn
                    figure;
                    plot(errorDistance)
                    title(['Error ',description]);
                    xlabel('Test Image');
                    ylabel('Error [meter]');

                    figure;
                    histogram(errorDistance)
                    title(['Histogram of the error ',description]);
                    xlabel('Error [meter]');
                    ylabel('Amount of frames');
                end
                %errorDistMSE = sum(errorDistance.^2)/size(errorDistance,1);
                errorDistMean = sum(errorDistance)/size(errorDistance,1);
                errorDistMax = max(errorDistance);
                errorDistMedian = median(errorDistance);
                errorPercentage = 100*size(find(errorDistance<2),1)/testDBSize;
    %             fprintf('\n--------------------------RESULT---------------------------------\n');
    %             fprintf('--INPUT:\n');
    %             fprintf(['For testDB nr.%d, using ',description,'\n'],testDB);
    %             fprintf('The edge detection thresholds are: Training=%.4f; Test=%.4f\n',edgeThresholdTraining,edgeThresholdTest);
    %             fprintf('The ConfMatCNN is %.4f, ConfMatObj is %.4f, ConfMatScene is %.4f\n',ConfMatCNN,ConfMatObj,ConfMatScene);
    %             fprintf('The width of room 91.68 is set to %.1f meter\n',widthRoom68);
    %             fprintf('\n--OUTPUT:\n');
    %             fprintf('Due to the edge detection, this amount of frames are dropped: Training=%.0f; Test=%.0f\n',size(TrainingToDelete,2),size(TestToDelete,2));
    %             fprintf('The mean of the error is %.4f meter, the median is %.4f meter and the maximal error is %.2f meter.\n',errorDistMean,errorDistMedian,errorDistMax);
    %             fprintf('In %.2f%% of the frames, the error is below 2 meter.\n',errorPercentage);
    %             fprintf('-----------------------------------------------------------------\n\n');
            %end
            %--------------------------------------------------------------------------

                switch i %locationMode
                        case 1
                            % No post-processing
                            finalResult_np(plotindex,:) = [ConfMatCNN ConfMatObj ConfMatScene errorDistMean errorDistMedian errorDistMax errorPercentage];
                        case 2
                            % Spatial Continuity filter
                            finalResult_sc(plotindex,:) = [ConfMatCNN ConfMatObj ConfMatScene errorDistMean errorDistMedian errorDistMax errorPercentage];
                        case 3
                            % Particle Filter
                            finalResult_pf(plotindex,:) = [ConfMatCNN ConfMatObj ConfMatScene errorDistMean errorDistMedian errorDistMax errorPercentage];
                end

            end
            plotindex = plotindex +1;

        end
    end
    finalResult_np = round(finalResult_np([4,2,1,3],[1:4,6:7]),2);
    finalResult_sc = round(finalResult_sc([4,2,1,3],[1:4,6:7]),2);
    finalResult_pf = round(finalResult_pf([4,2,1,3],[1:4,6:7]),2);
    finalResult_np_str = num2str(finalResult_np,4);
    finalResult_pf_str = num2str(finalResult_pf,4);
    finalResult_sc_str = num2str(finalResult_sc,4);
    
    
    if exist([datapath,'error2.mat'], 'file')
        save([datapath,'error2.mat'],'finalResult_np','finalResult_sc','finalResult_pf','finalResult_np_str','finalResult_sc_str','finalResult_pf_str','-append');
    else
        save([datapath,'error2.mat'],'finalResult_np','finalResult_sc','finalResult_pf','finalResult_np_str','finalResult_sc_str','finalResult_pf_str');
    end
else
    disp('Error not recalculated')
    load([datapath,'error2.mat'],'finalResult_np','finalResult_sc','finalResult_pf','finalResult_np_str','finalResult_sc_str','finalResult_pf_str');
end