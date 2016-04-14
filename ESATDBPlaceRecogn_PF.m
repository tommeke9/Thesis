clear all
close all
clc

TrainingCoordinates = makeTrainingCoordinates();
addpath data deps/matconvnet-1.0-beta16 data/ESAT-DB

%Run setup before! to compile matconvnet
%%
%------------------------VARIABLES-----------------------------------------
PlotOn = 1; %Plot Debugging figures

%WARNING: If change of testDB ==> RunCNN, RunConf, calcScenesTestDB, RunConfScene =1
testDB = 1; %Select the testDB: 1 (same day) or 2 (after ~2 months)

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

ConfMatCNN = 0.750; % Multiplied with the CNN feature CNN, and 1-ConfMatCNN is multiplied with the Scene Recogn Conf Matrix.
ConfMatObj = 0.125; % 
ConfMatScene = 0.125;
if ConfMatCNN+ConfMatObj+ConfMatScene ~=1
    error('Check the Confusion Matrix parameters.');
end

%Particle Filter
%FeatureDetectNoiseStDev = 200;  %Standard deviation on calculated difference of features
SpeedStDev = 2;                 %Standard deviation on calculated speed
Speed = 1;                      %speed of walking
RandPercentage = 0.1;           %Percentage of the particles to be randomized (1 = 100%)
N = 2500;                       %Amount of particles
PlotPF = 0;                     %1 = plot the PF for debugging & testing

locationMode = 3; %1 = No correction, 2 = Spatial Continuity, 3 = Particle Filtering

widthRoom68 = 3; %used to calculate the error

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
        T = load('test.mat');
        TestCoordinates = makeTestCoordinates();
    case 2
        T = load('test2.mat');
        TestCoordinates = makeTest2Coordinates();
end
testImg_original = T.img;
clear T
T = load('training.mat');
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
    if exist('data/lastFCesatDB.mat', 'file')
        save('data/lastFCesatDB.mat','lastFCtraining','-append');
    else
        save('data/lastFCesatDB.mat','lastFCtraining');
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
    if exist('data/lastFCesatDB.mat', 'file')
        save('data/lastFCesatDB.mat','lastFCtest','-append');
    else
        save('data/lastFCesatDB.mat','lastFCtest');
    end
    clear lastFCtemp res
    disp('CNN finished')
    %--------------------------------------------------------------------------
else
    disp('CNN not recalculated')
    load('lastFCesatDB.mat');
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
load('ESATsvm.mat');
disp('Scenes & SVM loaded')

%Retrain or Load the TrainingDB
if calcScenesTrainingDB
    disp('Recalculate Scenes for the trainingDB')
    
    for index = 1:trainingDBSize_original
        for i = 1:size(uniqueScenes,1)
            scoresTraining(index,i) = W(:,i)'*lastFCtraining_original(:,index) + B(i) ;
        end
    end
    
    if exist('data/ScenesEsatDB.mat', 'file')
        save('data/ScenesEsatDB.mat','scoresTraining','-append');
    else
        save('data/ScenesEsatDB.mat','scoresTraining');
    end
    disp('Scenes saved for the trainingDB')
else
    disp('Scenes for the TrainingDB not recalculated')
    load('data/ScenesEsatDB.mat','scoresTraining');
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
    
    if exist('data/ScenesEsatDB.mat', 'file')
        save('data/ScenesEsatDB.mat','scoresTest','bestScoreScene','bestScene','-append');
    else
        save('data/ScenesEsatDB.mat','scoresTest','bestScoreScene','bestScene');
    end
    disp('Scenes saved for the testDB')
else 
    disp('Scenes for the testDB not recalculated')
    load('data/ScenesEsatDB.mat','scoresTest','bestScoreScene','bestScene');
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
    if exist('data/confMatrix.mat', 'file')
        save('data/confMatrix.mat','confusionMatrixSceneRecogn','-append');
    else
        save('data/confMatrix.mat','confusionMatrixSceneRecogn');
    end
    disp('ConfusionMatrix of Scenes saved')
else
    disp('ConfusionMatrix of Scenes not recalculated')
    load('confMatrix.mat','confusionMatrixSceneRecogn');
end

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
    
    if exist('data/Objects.mat', 'file')
        save('data/Objects.mat','trainingObjectLocation','-append');
    else
        save('data/Objects.mat','trainingObjectLocation');
    end
    disp('Object locations saved for the trainingDB')
else
    disp('Object locations for the trainingDB not recalculated')
    load('data/Objects.mat','trainingObjectLocation');
end

if calcObjLocTest
    disp('Recalculate object locations for the testDB')
    delete(gcp('nocreate'))
    testObjectLocation = calc_object_locations( n_box_max, testImg_original );
    
    if exist('data/Objects.mat', 'file')
        save('data/Objects.mat','testObjectLocation','-append');
    else
        save('data/Objects.mat','testObjectLocation');
    end
    disp('Object locations saved for the testDB')
else
    disp('Object locations for the testDB not recalculated')
    load('data/Objects.mat','testObjectLocation');
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
   
    if exist('data/Objects.mat', 'file')
        save('data/Objects.mat','trainingObjectRecognition','bestScoreObject_training','objectNumber_training','objectName_training','-append');
    else
        save('data/Objects.mat','trainingObjectRecognition','bestScoreObject_training','objectNumber_training','objectName_training');
    end
    disp('Object recognition saved for the trainingDB')
else
    disp('Object recognition for the trainingDB not recalculated')
    load('data/Objects.mat','trainingObjectRecognition','bestScoreObject_training','objectNumber_training','objectName_training');
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
    
    if exist('data/Objects.mat', 'file')
        save('data/Objects.mat','testObjectRecognition','bestScoreObject_test','objectNumber_test','objectName_test','-append');
    else
        save('data/Objects.mat','testObjectRecognition','bestScoreObject_test','objectNumber_test','objectName_test');
    end
    disp('Object recognition saved for the testDB')
else
    disp('Object recognition for the testDB not recalculated')
    load('data/Objects.mat','testObjectRecognition','bestScoreObject_test','objectNumber_test','objectName_test');
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

    if exist('data/confMatrix.mat', 'file')
        save('data/confMatrix.mat','confusionMatrixObjects','-append');
    else
        save('data/confMatrix.mat','confusionMatrixObjects');
    end
else
    disp('ConfusionMatrix for the Object Recognition not recalculated')
    load('confMatrix.mat','confusionMatrixObjects');
end

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
    if exist('data/confMatrix.mat', 'file')
        save('data/confMatrix.mat','confusionMatrixCNNFeat','-append');
    else
        save('data/confMatrix.mat','confusionMatrixCNNFeat');
    end
else
    disp('ConfusionMatrix CNN features not recalculated')
    load('confMatrix.mat','confusionMatrixCNNFeat');
end

if PlotOn
    figure;
    imagesc(confusionMatrixCNNFeat)
    title('Confusion Matrix CNN features')
    ylabel('Training Image')
    xlabel('Test Image')
end
%--------------------------------------------------------------------------


%%
% %------------------------Confusion Matrix Edges---------------------
% if RunConfEdges
%     disp('Start calculating the confusion matrix for the Edges')
%     confusionMatrixEdges = zeros(trainingDBSize);
%     for index = 1:testDBSize
%         for i = 1:trainingDBSize
%             confusionMatrixEdges(i,index) = abs(threshOut_training(i) - threshOut_test(index));
%         end
% %         if rem(index,100)==0
% %                 fprintf('Confusion Calc. %d ~ %d of %d \n',index-99,index,testDBSize);
% %         end
%     end
%     if exist('data/confMatrix.mat', 'file')
%         save('data/confMatrix.mat','confusionMatrixEdges','-append');
%     else
%         save('data/confMatrix.mat','confusionMatrixEdges');
%     end
% else
%     disp('ConfusionMatrix Edges not recalculated')
%     load('confMatrix.mat','confusionMatrixEdges');
% end
% 
% if PlotOn
%     figure;
%     imagesc(confusionMatrixEdges)
%     title('Confusion Matrix Edges')
%     ylabel('Training Image')
%     xlabel('Test Image')
% end
% %--------------------------------------------------------------------------



%%
%------------------------Combine Confusion Matrices------------------------
disp('Start combining the confusion matrices')

%confusionMatrix = ConfMatCNN .* confusionMatrixCNNFeat + (1-ConfMatCNN) .* confusionMatrixSceneRecogn;
%confusionMatrix = confusionMatrixCNNFeat .* confusionMatrixSceneRecogn;
confusionMatrix = ConfMatCNN .* (confusionMatrixCNNFeat - min(min(confusionMatrixCNNFeat)))./max(max(confusionMatrixCNNFeat)) + ConfMatScene .* (confusionMatrixSceneRecogn - min(min(confusionMatrixSceneRecogn)))./max(max(confusionMatrixSceneRecogn)) + ConfMatObj .* (confusionMatrixObjects - min(min(confusionMatrixObjects)))./max(max(confusionMatrixObjects));
if PlotOn
    figure;
    imagesc(confusionMatrix)
    title('Combined Confusion Matrix')
    ylabel('Training Image')
    xlabel('Test Image')
end
%--------------------------------------------------------------------------

%%
%------------------------Select Lowest difference--------------------------
disp('Search lowest difference')
parfor index = 1:testDBSize
    [ResultValue(index),Result(index)] = min(confusionMatrix(:,index));
%     if rem(index,100)==0
%             fprintf('%d ~ %d of %d \n',index-99,index,testDBSize);
%     end
end
if PlotOn
    figure;
    plot(Result,'g')
    hold on
end
%%
%------------------------Spatial Continuity check--------------------------
d = 2; % Length of evaluation window
epsilon = 3;
for index = d:testDBSize
    P(index) = 1;
    for u = index-d+2:index
        if abs(ResultValue(u-1)-ResultValue(u)) > epsilon
            P(index) = 0;
            break;
        end
    end
end

%HOLD THE PREVIOUS VALUE IF P=0
ResultSC(1) = Result(1);
for index = 2:testDBSize
    if P(index) == 1
        ResultSC(index) = Result(index);
    else
        ResultSC(index) = ResultSC(index-1);
    end
end
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
    vPF = VideoWriter('data/VideoPF.avi');
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
    fprintf('\n--------------------------RESULT---------------------------------\n');
    fprintf('--INPUT:\n');
    fprintf(['For testDB nr.%d, using ',description,'\n'],testDB);
    fprintf('The edge detection thresholds are: Training=%.4f; Test=%.4f\n',edgeThresholdTraining,edgeThresholdTest);
    fprintf('The ConfMatCNN is %.4f, ConfMatObj is %.4f, ConfMatScene is %.4f\n',ConfMatCNN,ConfMatObj,ConfMatScene);
    fprintf('The width of room 91.68 is set to %.1f meter\n',widthRoom68);
    fprintf('\n--OUTPUT:\n');
    fprintf('Due to the edge detection, this amount of frames are dropped: Training=%.0f; Test=%.0f\n',size(TrainingToDelete,2),size(TestToDelete,2));
    fprintf('The mean of the error is %.2f meter, and the maximal error is %.2f meter.\n',errorDistMean,errorDistMax);
    fprintf('-----------------------------------------------------------------\n\n');
end
%--------------------------------------------------------------------------

%%
%------------------------------Show traject on map--------------------------
plotHeight = 1;
if locationMode == 3
    plotHeight = 2;
end
if PlotRoute
    figure('units','normalized','outerposition',[0 0 1 1]);
    [X,map] = imread('floorplan.gif');
    if ~isempty(map)
        Im = ind2rgb(X,map);
    end
    %imshow(Im)
    %hold on;
    v = VideoWriter('data/newfile.avi');
    open(v)
    for i=1:testDBSize
        subplot(plotHeight,3,1)
        imshow(Im)
        hold on;
        plot(testLocations(i,1),testLocations(i,2),'or','MarkerSize',5,'MarkerFaceColor','r')
        text(500,570,['current test-photo: ' num2str(i)],'Color','r')
        hold off;
        if bestScoreScene(i) > 1
            title(['Looks like a ',uniqueScenes(bestScene(i)),' with score ',bestScoreScene(i)],'interpreter','none','color',[0,0,0])
        elseif bestScoreScene(i) > 0.3
            title(['Looks like a ',uniqueScenes(bestScene(i)),' with score ',bestScoreScene(i)],'interpreter','none','color',[1-bestScoreScene(i),1-bestScoreScene(i),1-bestScoreScene(i)])
        end
        
        subplot(plotHeight,3,2)
        imshow(testImg(:,:,:,i));
        title(['Test image: ',num2str(i)])
        
        
        
        
        if locationMode == 1
            subplot(plotHeight,3,3)
            imshow(trainingImg(:,:,:,Result(i)));
            title(['(Original method) Training image: ',num2str(Result(i))])
        elseif locationMode == 2
            subplot(plotHeight,3,3)
            imshow(trainingImg(:,:,:,ResultSC(i)));
            title(['(Sequential Filter method) Training image: ',num2str(ResultSC(i))])
        elseif locationMode == 3
            subplot(plotHeight,3,3)
            imshow(trainingImg(:,:,:,ResultPF(i)));
            title(['(PF method) Training image: ',num2str(ResultPF(i))])
            
            subplot(plotHeight,3,4:6)
            %histogram(storedParticles(:,i));
            scatter(storedParticles(:,i),ones(size(storedParticles(:,i),1),1));
            hold on
            line([mode(storedParticles(:,i)) mode(storedParticles(:,i))], [0.5 1.5], 'color','r','linewidth',2);
            hold off
            axis([0 trainingDBSize 0 inf])
            %title('Particles histogram')
            title(['Particles filter with N = ',num2str(N),'; RandPercentage = ',num2str(RandPercentage),'; SpeedStDev = ',num2str(SpeedStDev),'; FeatureDetectNoiseStDev = ',num2str(FeatureDetectNoiseStDev)]);
            xlabel('Training Image');
            %ylabel('Amount of particles');
        end
        
        frame = getframe(gcf);
        writeVideo(v,frame)
        %pause(.02);
    end
    close(v);
end


