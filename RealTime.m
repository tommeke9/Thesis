clear all
close all
clc

TrainingCoordinates = makeTrainingCoordinates();
addpath data deps/matconvnet-1.0-beta16 data/ESAT-DB

%Run setup before! to compile matconvnet
%%
%------------------------VARIABLES-----------------------------------------


%WARNING: If new testDB ==> RunCNN, RunConfCNN, calcScenesTestDB, RunConfScene, calcObjLocTest, calcObjRecTest, RunConfObjects =1

datapath = 'data/test/realtime/';
lastFClayer = 13;

%WARNING: If change of edgeThreshold ==> Put RunConfScene AND RunConfCNN AND RunConfObjects to 1
edgeThresholdTraining = 0;%0.05;
edgeThresholdTest = 0.075;%0.05;

%-------Training
RunCNN_training = 0; %1 = run the CNN for the training (only with RunCNN = 1)

%Scene Recognition
calcScenesTrainingDB = 0;   %1 if recalc of the scenes for the trainingDB is necessary.

%Object Localisation
calcObjLocTraining = 0;

%Object Recognition
calcObjRecTraining = 0;
%-----

RunObjects_test = 0;% 1= run the object localization & recognition during the testfase

PlotRoute = 1;  %1 = plot the route on a floorplan



%Object Localisation
n_box_max = 5; % Max amount of boxes to be used for object recognition

%Object Recognition
n_labels_max = 5; %Max amount of recognized objects per box

ConfMatCNN = 0.600; % Multiplied with the CNN feature CNN, and 1-ConfMatCNN is multiplied with the Scene Recogn Conf Matrix.
ConfMatObj = 0.100; % 
ConfMatScene = 0.300;
if ConfMatCNN+ConfMatObj+ConfMatScene ~=1
    error('Check the Confusion Matrix parameters.');
end

%Spatial Continuity check
d = 40; % Length of evaluation window
epsilon = 50;% maximal jumps of trainingframes in this evaluation window

%Particle Filter
%FeatureDetectNoiseStDev = 200;  %Standard deviation on calculated difference of features
SpeedStDev = 2;                 %Standard deviation on calculated speed
Speed = 1;                      %speed of walking
RandPercentage = 0.1;           %Percentage of the particles to be randomized (1 = 100%)
N = 2500;                       %Amount of particles
PlotPF = 0;                     %1 = plot the PF for debugging & testing

locationMode = 3; %1 = No correction, 2 = Spatial Continuity, 3 = Particle Filtering
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
T = load('data/ESAT-DB/mat/training.mat');
trainingImg_original = T.img;
clear T
disp('DB loaded')

% Define the sizes of the DB
trainingDBSize_original = size(trainingImg_original,4);

%Setup MatConvNet
delete(gcp('nocreate'))
run deps/matconvnet-1.0-beta16/matlab/vl_setupnn;

%%
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------TRAINING----------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%--------------------------Edge  Detection---------------------------------
%Leave the images out of the training & test if the amount of edges is below a
%specific treshold
%Parfor gives speedimprovement for trainingDB: 17sec -> 10sec

trainingImg = trainingImg_original;

TrainingToDelete = [];

if edgeThresholdTraining~=0
    parfor index = 1:trainingDBSize_original
        [~,threshOut_training(index)] = edge(rgb2gray(trainingImg(:,:,:,index)));
        if threshOut_training(index) < edgeThresholdTraining
            TrainingToDelete = [TrainingToDelete,index];
        end
    end
    trainingImg(:,:,:,TrainingToDelete(:)) = [];
    TrainingCoordinates(TrainingToDelete(:),:) = [];
end

% Define the sizes of the new DB
trainingDBSize = size(trainingImg,4);
%--------------------------------------------------------------------------

% ------------load and preprocess the images---------------------------------
if RunCNN_training
    disp('Normalization')
    imSize = net.meta.normalization.imageSize(1:2);

    for index = 1:trainingDBSize_original
        im_temp = single(trainingImg_original(:,:,:,index)) ; % note: 0-255 range
        im_temp = imresize(im_temp, imSize) ;
        trainingImgNorm(:,:,:,index) = im_temp - averageImage ;
    end
    clear im_temp index imSize

    disp('Normalization finished')
end
%--------------------------------------------------------------------------

    
% ---------------------------------Run CNN---------------------------------
if RunCNN_training
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
else
    disp('Only CNN of the testDB recalculated')
    load([datapath,'lastFCesatDB.mat'],'lastFCtraining');
end

disp('CNN finished')
%--------------------------------------------------------------------------

lastFCtraining_original = lastFCtraining;

lastFCtraining(:,TrainingToDelete(:)) = [];
%-------------------------------Scene Recognition--------------------------
%GOAL: Save for every test image the scores for every scenetype. (Thus
%a score for every scene in the testDB) This will be added to the
%ConfusionMatrix and used for the localisation.

disp('Load scenes & SVM')
load('data/ESATsvm.mat');
disp('Scenes & SVM loaded')

%Retrain or Load the TrainingDB
if calcScenesTrainingDB
    disp('Recalculate Scenes for the trainingDB')
  
    for index = 1:trainingDBSize_original
        for i = 1:size(uniqueScenes,1)
            scoresScenesTraining(index,i) = W(:,i)'*lastFCtraining_original(:,index) + B(i) ;
        end
    end
    
    if exist([datapath,'ScenesEsatDB.mat'], 'file')
        save([datapath,'ScenesEsatDB.mat'],'scoresScenesTraining','-append');
    else
        save([datapath,'ScenesEsatDB.mat'],'scoresScenesTraining');
    end
    disp('Scenes saved for the trainingDB')
else
    disp('Scenes for the TrainingDB not recalculated')
    load([datapath,'ScenesEsatDB.mat'],'scoresScenesTraining');
end
scoresScenesTraining(TrainingToDelete(:),:) = [];
%--------------------------------------------------------------------------

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




trainingObjectLocation(:,:,TrainingToDelete(:)) = [];

trainingObjectRecognition(:,:,:,TrainingToDelete(:)) = [];
bestScoreObject_training(:,TrainingToDelete(:)) = [];
objectName_training(:,TrainingToDelete(:)) = [];
%--------------------------------------------------------------------------
delete(gcp('nocreate'))
disp('------Training finished--------')














%%
if ~RunObjects_test
    ConfMatScene = 1-ConfMatCNN;
end
switch locationMode
    case 2 %Spatial Continuity
        lastResults = zeros(d,1);
    case 3 %Particle Filter
        %Initialize particles
        particles = round(rand(N,1)*(trainingDBSize-1)+1);
        w=ones(N,1)./N;
        
end


%Startup plot
if PlotRoute
    plotHeight = 1;
    if locationMode == 3
        plotHeight = 2;
    end
    switch locationMode
        case 1
            % No post-processing
            description = 'no post-processing';
        case 2
            % Spatial Continuity filter
            description = ['the spatial continuity filter with: epsilon=' num2str(epsilon) '; d=' num2str(d)];
        case 3
            % Particle Filter
            description = ['the particle filter with N=',num2str(N),'; Speed=',num2str(Speed),'; RandPercentage=',num2str(RandPercentage),'; SpeedStDev=',num2str(SpeedStDev)];
    end

    figure('units','normalized','outerposition',[0 0 1 1]);
    [X,map] = imread('floorplan.gif');
    if ~isempty(map)
        floorplan = ind2rgb(X,map);
    end
    subplot(plotHeight,3,1)
    imshow(floorplan)
    hold on;
end

cam = webcam(1);

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
    
    %Scene recognition
    scoresTest = zeros(1,size(uniqueScenes,1));
    for i = 1:size(uniqueScenes,1)
        scoresTest(i) = W(:,i)'*lastFCtest + B(i) ;
    end
    [bestScoreScene, bestScene] = max(scoresTest(:)) ;
    
    %objectLocalisation & recognition
    if RunObjects_test
        testObjectLocation = calc_object_locations( n_box_max, testImg );
        testObjectRecognition = calc_object_recognition( testImg, testObjectLocation, net, n_labels_max );
        
        bestScoreObject_test = squeeze(testObjectRecognition(1,2,:,:)) ; % bestScoreObject_test(Frame_number,img_Number)
        objectNumber_test = squeeze(testObjectRecognition(1,1,:,:)) ; % objectNumber_test(Frame_number,img_Number)
        objectName_test = reshape(net.meta.classes.description(objectNumber_test(:)),size(objectNumber_test));
    end
    
    
    %Make confusion-array
    if locationMode == 1 || locationMode == 2
        %make confusion array
        confusionArrayCNN = zeros(1,trainingDBSize);
        confusionArrayScene = zeros(1,trainingDBSize);
        confusionArrayObjects = ones(1,trainingDBSize).*n_box_max;
        for i = 1:trainingDBSize
            %CNN
            confusionArrayCNN(i) = norm(lastFCtest(:)-lastFCtraining(:,i));
            
            %Scenes
            confusionArrayScene(i) = norm(scoresTest-scoresScenesTraining(i,:));
            
            %objects
            if RunObjects_test
                test = objectNumber_test(:);
                for q = 1:n_box_max
                    for z = 1:size(test,1)
                        if objectNumber_training(q,i) == test(z)
                            confusionArrayObjects(i) = confusionArrayObjects(i) - 1;
                            test(z) = [];
                            break
                        end
                    end
                end
            end
        end
        confusionArrayCNN = (confusionArrayCNN - min(min(confusionArrayCNN)))./max(max(confusionArrayCNN));
        confusionArrayScene = (confusionArrayScene - min(min(confusionArrayScene)))./max(max(confusionArrayScene));
        confusionArrayObjects = (confusionArrayObjects - min(min(confusionArrayObjects)))./max(max(confusionArrayObjects));
        
        %combine the confusionarrays
        if RunObjects_test
            confusionArray = ConfMatCNN .* confusionArrayCNN + ConfMatScene .* confusionArrayScene + ConfMatObj .* confusionArrayObjects;
        else
            confusionArray = ConfMatCNN .* confusionArrayCNN + ConfMatScene .* confusionArrayScene;
        end
        
        %select lowest value
        [ResultValue,Result] = min(confusionArray(:));
    end
    
    switch locationMode
        case 1 %No post-processing
            testLocation = [TrainingCoordinates(Result,1),TrainingCoordinates(Result,2)];
            
        case 2 %Spatial Filtering
            %save the last 'd' results
            if loopIndex >= d
                lastResults(1) = [];
            end
            lastResults(min(loopIndex,d)) = Result;
            
            %Spatial check
            P = 1;
            for u = 2:min(loopIndex,d)
                if abs(lastResults(u-1)-lastResults(u)) > epsilon
                    P = 0;
                    break;
                end
            end
            
            %Only motion model if P=0
            if P == 1
                ResultSC = Result;
            else
                if loopIndex == 1
                    ResultSC = Result;
                else
                    ResultSC = ResultSC+1;
                end
                
                if ResultSC>trainingDBSize
                    ResultSC = trainingDBSize;
                end
            end
 
            testLocation = [TrainingCoordinates(ResultSC,1),TrainingCoordinates(ResultSC,2)];
            
            
        case 3 %Particle Filtering
            %only check trainingimages if particles are present ==> smaller
            %confusion array
            
            %Edge detect
            [~,threshOut_test] = edge(rgb2gray(testImg));
            if threshOut_test < edgeThresholdTest
                TestToDelete = 1;
            else
                TestToDelete = 0;
            end
    
            if ~TestToDelete
                %make confusion array
                confusionArrayCNN = zeros(1,trainingDBSize);
                confusionArrayScene = zeros(1,trainingDBSize);
                confusionArrayObjects = ones(1,trainingDBSize).*n_box_max;
                for i = unique(particles)'
                    %CNN
                    confusionArrayCNN(i) = norm(lastFCtest(:)-lastFCtraining(:,i));

                    %Scenes
                    confusionArrayScene(i) = norm(scoresTest-scoresScenesTraining(i,:));

                    %objects
                    if RunObjects_test
                        test = objectNumber_test(:);
                        for q = 1:n_box_max
                            for z = 1:size(test,1)
                                if objectNumber_training(q,i) == test(z)
                                    confusionArrayObjects(i) = confusionArrayObjects(i) - 1;
                                    test(z) = [];
                                    break
                                end
                            end
                        end
                    end
                end
                confusionArrayCNN = (confusionArrayCNN - min(min(confusionArrayCNN)))./max(max(confusionArrayCNN));
                confusionArrayScene = (confusionArrayScene - min(min(confusionArrayScene)))./max(max(confusionArrayScene));
                confusionArrayObjects = (confusionArrayObjects - min(min(confusionArrayObjects)))./max(max(confusionArrayObjects));

                %combine the confusionarrays
                if RunObjects_test
                    confusionArray = ConfMatCNN .* confusionArrayCNN + ConfMatScene .* confusionArrayScene + ConfMatObj .* confusionArrayObjects;
                else
                    confusionArray = ConfMatCNN .* confusionArrayCNN + ConfMatScene .* confusionArrayScene;
                end
                
                %Create weights using the normal distribution pdf & normalize
                FeatureDetectNoiseStDev = sqrt(var(confusionArray(unique(particles))));
                w=ones(N,1)./N;
                w = w.*((1/(sqrt(2*pi)*FeatureDetectNoiseStDev)*exp(-(  confusionArray(particles(:))).^2/(2*FeatureDetectNoiseStDev^2))))';
                w = w/sum(w);
                
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

            if ~TestToDelete
                %Randomize a specific percentage to avoid locked particles
                particles(round(rand(ceil(N*RandPercentage),1)*(N-1)+1)) = round(rand(ceil(N*RandPercentage),1)*(trainingDBSize-1)+1);
            end
            %Keep result of the PF
            ResultPF = mode(particles);
            
            %PF
            testLocation = [TrainingCoordinates(ResultPF,1),TrainingCoordinates(ResultPF,2)];
    end
    
    toc
    
    %Show location on map
    if PlotRoute
        plot(testLocation(1),testLocation(2),'or','MarkerSize',5,'MarkerFaceColor','r')
        hold off;

        subplot(plotHeight,3,2)
        imshow(testImg);
        
        if bestScoreScene > 1
            title(['Input image looks like a ',uniqueScenes(bestScene),' with score ',bestScoreScene],'interpreter','none','color',[0,0,0])
        elseif bestScoreScene > 0.3
            title(['Input image looks like a ',uniqueScenes(bestScene),' with score ',bestScoreScene],'interpreter','none','color',[1-bestScoreScene,1-bestScoreScene,1-bestScoreScene])
        else
            title('I do not know what it looks like...','interpreter','none','color',[0,0,0])
        end

        switch locationMode
            case 1
                subplot(plotHeight,3,3)
                imshow(trainingImg(:,:,:,Result));
                title(['(Original method) Training image: ',num2str(Result)])
            case 2
                subplot(plotHeight,3,3)
                imshow(trainingImg(:,:,:,ResultSC));
                title(['(Sequential Filter method) Training image: ',num2str(ResultSC)])
            case 3
                subplot(plotHeight,3,3)
                imshow(trainingImg(:,:,:,ResultPF));
                title(['(PF method) Training image: ',num2str(ResultPF)])

                subplot(plotHeight,3,4:6)
                histogram(particles(:),round(trainingDBSize/10));
                hold on
                axis([0 trainingDBSize 0 inf])
                title(description);
                xlabel('Training Image');
                ylabel('Amount of particles');
                line([ResultPF ResultPF], [0 1000], 'color','r');
                hold off

        end
    end
end