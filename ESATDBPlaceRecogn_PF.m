clear all
close all
clc

ImageCoordinates = makeTrainingCoordinates();
addpath data deps/matconvnet-1.0-beta16 data/ESAT-DB

%Run setup before! to compile matconvnet
%Variables:
lastFClayer = 31;
edgeThresholdTraining = 0.07;
edgeThresholdTest = 0.05;
RunCNN = 0; %1 = run the CNN, 0 = Load the CNN
RunConf = 0; %1 = recalc the Conf. matrix, 0 = Load the Conf. Matrix
PlotRoute = 0; %1 = plot the route on a floorplan

%Scene Recognition
calcScenesTrainingDB = 0; %1 if recalc of the scenes for the trainingDB is necessary.
calcScenesTestDB = 0; %1 if recalc of the scenes for the testDB is necessary.
RunConfScene = 0; %1 = recalc the Conf. matrix for the Scene Recognition, 0 = Load the Conf. Matrix

ConfMatCNN = 0.875; % Multiplied with the CNN feature CNN, and 1-ConfMatCNN is multiplied with the Scene Recogn Conf Matrix.

%Variables for PF
FeatureDetectNoiseStDev = 200; %Standard deviation on calculated difference of features
SpeedStDev = 2; %Standard deviation on calculated speed
Speed = 1; %speed of walking
RandPercentage = 0.1; %Percentage of the particles to be randomized (1 = 100%)
N = 2500; %Amount of particles
PlotPF = 0; %1 = plot the PF for debugging & testing

locationMode = 3; %1 = No correction, 2 = Spatial Continuity, 3 = Particle Filtering

% load the pre-trained CNN
net = load('imagenet-vgg-verydeep-16.mat') ;

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
delete(gcp('nocreate'))
run deps/matconvnet-1.0-beta16/matlab/vl_setupnn;

%--------------------------Edge  Detection---------------------------------
%Leave the images out of the training & test if the amount of edges is below a
%specific treshold

toDelete = [];
parfor index = 1:trainingDBSize
    [~,threshOut] = edge(rgb2gray(trainingImg(:,:,:,index)));
    if threshOut < edgeThresholdTraining
        toDelete = [toDelete,index];
    end
end
trainingImg(:,:,:,toDelete(:)) = [];
ImageCoordinates(toDelete(:),:) = [];


toDelete = [];
parfor index = 1:testDBSize
    [~,threshOut] = edge(rgb2gray(testImg(:,:,:,index)));
    if threshOut < edgeThresholdTest
        toDelete = [toDelete,index];
    end
end
testImg(:,:,:,toDelete(:)) = [];

%--------------------------------------------------------------------------
% Define the sizes of the new DB
trainingDBSize = size(trainingImg,4);
testDBSize = size(testImg,4);


if RunCNN



    
    % ------------load and preprocess the images---------------------------------
    disp('Normalization')
    parfor index = 1:trainingDBSize
        im_temp = single(trainingImg(:,:,:,index)) ; % note: 0-255 range
        im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
        trainingImgNorm(:,:,:,index) = im_temp - net.normalization.averageImage ;
    end
    clear im_temp index
    parfor index = 1:testDBSize
        im_temp = single(testImg(:,:,:,index)) ; % note: 0-255 range
        im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
        testImgNorm(:,:,:,index) = im_temp - net.normalization.averageImage ;
    end
    clear im_temp index
    disp('Normalization finished')
    %--------------------------------------------------------------------------


    % ---------------------------------Run CNN---------------------------------
    disp('Run CNN')
    delete(gcp('nocreate'))
    for index = 1:trainingDBSize
        res = vl_simplenn(net, trainingImgNorm(:,:,:,index)) ;
        
        lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
        lastFCtraining(:,index) = lastFCtemp(:);
        
        if rem(index,100)==0
            fprintf('training %d ~ %d of %d \n',index-99,index,trainingDBSize);
        end
    end
    save('data/lastFCesatDB.mat','lastFCtraining','-append');
    clear lastFCtemp index res
    
    %Same for testDB
    for index = 1:testDBSize
        res = vl_simplenn(net, testImgNorm(:,:,:,index)) ;
        
        lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
        lastFCtest(:,index) = lastFCtemp(:);
        
        if rem(index,100)==0
            fprintf('test %d ~ %d of %d \n',index-99,index,testDBSize);
        end
    end
    save('data/lastFCesatDB.mat','lastFCtest','-append');
    clear lastFCtemp res
    disp('CNN finished')
    %--------------------------------------------------------------------------
else
    disp('CNN not recalculated')
    load('lastFCesatDB.mat');
end


%-------------------------------Scene Recognition--------------------------
%GOAL: Save for every test image the scores for every scenetype. (Thus
%a score for every scene in the testDB) This will be added to the
%ConfusionMatrix and used for the localisation.

disp('Load scenes')
load('newDB.mat','sceneTypes')
uniqueScenes = unique(sceneTypes);
clear sceneTypes
disp('Scenes loaded')

%Retrain or Load the TrainingDB
if calcScenesTrainingDB
    disp('Recalculate Scenes for the trainingDB')
    scoresTraining = Train_scenes_ESATDB(trainingImg,net,lastFClayer);
    save('data/ScenesEsatDB.mat','scoresTraining','-append');
    disp('Scenes saved for the trainingDB')
else
    disp('Scenes for the TrainingDB not recalculated')
    load('data/ScenesEsatDB.mat','scoresTraining');
end

%Retrain or Load the TestDB
if calcScenesTestDB
    disp('recalculate scenes for the testDB')
    scoresTest = Train_scenes_ESATDB(testImg,net,lastFClayer);
    
    %Save the best scene with the score
    for index = 1:testDBSize
        [bestScoreScene(index), bestScene(index)] = max(scoresTest(index,:)) ;
    end
    save('data/ScenesEsatDB.mat','scoresTest','bestScoreScene','bestScene','-append');
    disp('Scenes saved for the testDB')
else 
    disp('Scenes for the testDB not recalculated')
    load('data/ScenesEsatDB.mat','scoresTest','bestScoreScene','bestScene');
end

%Make a temporary confusionMatrix for the scene-recognition
if RunConfScene
    disp('Start to compare scenes')
    confusionMatrixSceneRecogn = zeros(trainingDBSize);
    for index = 1:testDBSize
        for i = 1:trainingDBSize
            confusionMatrixSceneRecogn(i,index) = norm(scoresTest(index,:)-scoresTraining(i,:));
        end
        if rem(index,100)==0
                fprintf('Scene Calc. %d ~ %d of %d \n',index-99,index,testDBSize);
        end
    end
    save('data/confMatrix.mat','confusionMatrixSceneRecogn','-append');
    disp('ConfusionMatrix of Scenes saved')
else
    disp('ConfusionMatrix of Scenes not recalculated')
    load('confMatrix.mat','confusionMatrixSceneRecogn');
end

figure;
imagesc(confusionMatrixSceneRecogn)
title('Confusion Matrix Scene Recognition')
xlabel('Training Image')
ylabel('Test Image')

%--------------------------------------------------------------------------



%------------------------Confusion Matrix CNN Features---------------------
if RunConf
    disp('Start tests')
    confusionMatrixCNNFeat = zeros(trainingDBSize);
    parfor index = 1:testDBSize
        for i = 1:trainingDBSize
            confusionMatrixCNNFeat(i,index) = norm(lastFCtest(:,index)-lastFCtraining(:,i));
        end
        if rem(index,100)==0
                fprintf('Confusion Calc. %d ~ %d of %d \n',index-99,index,testDBSize);
        end
    end
    save('data/confMatrix.mat','confusionMatrixCNNFeat','-append');
else
    disp('ConfusionMatrix not recalculated')
    load('confMatrix.mat');
end

figure;
imagesc(confusionMatrixCNNFeat)
title('Confusion Matrix CNN features')
xlabel('Training Image')
ylabel('Test Image')

%--------------------------------------------------------------------------





%------------------------Combine Confusion Matrices------------------------
disp('Start combining the confusion matrices')

%confusionMatrix = ConfMatCNN .* confusionMatrixCNNFeat + (1-ConfMatCNN) .* confusionMatrixSceneRecogn;
%confusionMatrix = confusionMatrixCNNFeat .* confusionMatrixSceneRecogn;
confusionMatrix = ConfMatCNN .* (confusionMatrixCNNFeat - min(min(confusionMatrixCNNFeat)))./max(max(confusionMatrixCNNFeat)) + (1-ConfMatCNN) .* (confusionMatrixSceneRecogn - min(min(confusionMatrixSceneRecogn)))./max(max(confusionMatrixSceneRecogn));
figure;
imagesc(confusionMatrix)
title('Combined Confusion Matrix')
xlabel('Training Image')
ylabel('Test Image')

%--------------------------------------------------------------------------

%------------------------Select Lowest difference--------------------------
disp('Search lowest difference')
parfor index = 1:testDBSize
    [ResultValue(index),Result(index)] = min(confusionMatrix(:,index));
%     if rem(index,100)==0
%             fprintf('%d ~ %d of %d \n',index-99,index,testDBSize);
%     end
end
figure;
plot(Result,'g')
hold on

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
Resultnew(1) = Result(1);
for index = 2:testDBSize
    if P(index) == 1
        Resultnew(index) = Result(index);
    else
        Resultnew(index) = Resultnew(index-1);
    end
end
plot(Resultnew,'r')
%plot(Resultnew-Result,'g')
hold off
title(['Green = initial, Red = after Spatial Continuity Check with: epsilon = ' num2str(epsilon) '; d = ' num2str(d)])
xlabel('test images')
ylabel('training images')



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






%To delete/fix
FeatureDetectNoiseStDev = max(range(confusionMatrix))/2;










%-------------------------------Particle Filter--------------------------
%Initialize particles
particles = round(rand(N,1)*(trainingDBSize-1)+1);

if PlotPF
    figure;
end
for index = 1:testDBSize
    
    %Create weights using the normal distribution pdf & normalize
    w=ones(N,1)./N;
    w = w.*(1/(sqrt(2*pi)*FeatureDetectNoiseStDev)*exp(-(  confusionMatrix(particles(:),index)).^2/(2*FeatureDetectNoiseStDev^2)));
    w = w/sum(w);
    
    if PlotPF
        %plot the particles
        subplot(3,3,1:3);
        stem(particles(:),w*10000);
        axis([0 trainingDBSize 0 inf])
        title('Particle weights');
        xlabel('Training Image');
        ylabel('Weight (*10000)');
    end
    
    %Resample the particles = leave out the unlikely particles
    u = rand(N,1);
    wc = cumsum(w);
    [~,ind1] = sort([u;wc]);
    ind=find(ind1<=N)-(0:N-1)';
    particles=particles(ind);
    
    
    if PlotPF
        subplot(3,3,4);
        imshow(testImg(:,:,:,index));
        title(['Test Image ',num2str(index)]);

        subplot(3,3,5);
        imshow(trainingImg(:,:,:,mode(particles)));
        title(['Training Image ',num2str(mode(particles))]);
        
        %probability(index) = sum(abs(w - mean(w)).^2)/N;%sum(w > mean(w));
        
%         subplot(3,3,6);
%         plot(probability);
%         axis([0 testDBSize 0 inf])
%         title('Pobability of correctness');
%         xlabel('testImage');
%         ylabel('Prob');
        
        subplot(3,3,7:9);
        histogram(particles);
        hold on
        axis([0 trainingDBSize 0 inf])
        title('Particle histogram');
        xlabel('Training Image');
        ylabel('Amount of particles');
    end
    
    %motion model
    particles = round(particles + Speed + SpeedStDev*randn(size(particles)));
    particles(particles<=0)=1;
    particles(particles>=trainingDBSize)=trainingDBSize;
    
    %Randomize a specific percentage to avoid locked particles
    particles(round(rand(ceil(N*RandPercentage),1)*(N-1)+1)) = round(rand(ceil(N*RandPercentage),1)*(trainingDBSize-1)+1);
    
    %Keep result of the PF
    ResultPF(index) = mode(particles);
    
    if PlotPF
        line([mode(particles) mode(particles)], [0 1000], 'color','r');
        hold off
        
        drawnow
    end
    storedParticles(:,index) = particles;
end
plot(Result,'g')
hold on
plot(ResultPF,'r')
hold off
title({['Green = initial, Red = after Particle Filtering with N=',num2str(N),'; Speed=',num2str(Speed)],['RandPercentage=',num2str(RandPercentage),'; SpeedStDev=',num2str(SpeedStDev),'; FeatureDetectNoiseStDev=',num2str(FeatureDetectNoiseStDev)]});
xlabel('Test Image');
ylabel('Training Image');



%------------------------------Show traject on map--------------------------
plotHeight = 1;
if locationMode == 1
    testLocations = [ImageCoordinates(Result(1,:),1),ImageCoordinates(Result(1,:),2)];
elseif locationMode == 2
    testLocations = [ImageCoordinates(Resultnew(1,:),1),ImageCoordinates(Resultnew(1,:),2)];
elseif locationMode == 3
    testLocations = [ImageCoordinates(ResultPF(1,:),1),ImageCoordinates(ResultPF(1,:),2)];
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
    tic
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
            imshow(trainingImg(:,:,:,Resultnew(i)));
            title(['(Sequential Filter method) Training image: ',num2str(Resultnew(i))])
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
    toc
    close(v);
end


