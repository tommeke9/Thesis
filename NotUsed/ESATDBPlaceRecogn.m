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
PlotRoute = 1; %1 = plot the route on a floorplan

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
    % load the pre-trained CNN
    net = load('imagenet-vgg-verydeep-16.mat') ; %TO BE CHANGED TO VGG Places2



    
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
    save('data/lastFCesatDB.mat','lastFCtraining');
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
    parfor index = 1:testDBSize
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
parfor index = 1:testDBSize
    [ResultValue(index),Result(index)] = min(confusionMatrix(:,index));
%     if rem(index,100)==0
%             fprintf('%d ~ %d of %d \n',index-99,index,testDBSize);
%     end
end
figure;
plot(Result,'g')
hold on

%-------------------------------Spatial Continuity check--------------------------
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


%-------------------------------Scene Recognition--------------------------
%Done in sceneRecognitionESATDB_testonly.m ==> implement here?
%Takes very long...


%------------------------------Show traject on map--------------------------
testLocations = [ImageCoordinates(Resultnew(1,:),1),ImageCoordinates(Resultnew(1,:),2)];
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
        subplot(1,3,1)
        imshow(Im)
        hold on;
        plot(testLocations(i,1),testLocations(i,2),'or','MarkerSize',5,'MarkerFaceColor','r')
        text(500,570,['current test-photo: ' num2str(i)],'Color','r')
        hold off;
        
        subplot(1,3,2)
        imshow(testImg(:,:,:,i));
        title(['Test image: ',num2str(i)])
        
        subplot(1,3,3)
        imshow(trainingImg(:,:,:,Resultnew(i)));
        title(['Training image: ',num2str(Resultnew(i))])
        
        
        frame = getframe(gcf);
        writeVideo(v,frame)
        %pause(.02);
    end
    toc
    close(v);
end


