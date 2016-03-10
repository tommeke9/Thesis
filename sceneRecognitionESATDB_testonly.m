clear all
close all
clc

calcScenes = 0; %1 if recalc of the scenes for the testDB is necessary.



disp('loading ESAT DB')
T = load('test.mat');
testImg = T.img;
clear T
disp('DB loaded')
load('scenes.mat') %To be deleted...
testDBSize = size(testImg,4);

if calcScenes
    %Setup MatConvNet
    run matconvnet-1.0-beta16/matlab/vl_setupnn;

    % load the pre-trained CNN
    net = load('imagenet-vgg-verydeep-16.mat') ; %TO BE CHANGED TO VGG Places2

    load('svm.mat');
    lastFClayer = 31;
    for index = 1:testDBSize
            %normalize
            im_temp = single(testImg(:,:,:,index)) ; % note: 0-255 range
            im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
            testImgNorm(:,:,:,index) = im_temp - net.normalization.averageImage ;

            %Run CNN
            res = vl_simplenn(net, testImgNorm(:,:,:,index)) ;
            lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
            lastFCTest = lastFCtemp(:);

            %Select correct scene
            for i = 1:size(uniqueScenes,1)
                scores(:,i) = W(:,i)'*lastFCTest + B(i) ; %changed the * to . 
            end
            [bestScore(index), best(index)] = max(scores) ;
    end

    save('data/ScenesEsatDB.mat','bestScore','best');
else 
    disp('Scenes not recalculated')
    load('data/ScenesEsatDB.mat','bestScore','best');
end

figure('units','normalized','outerposition',[0 0 1 1]);
for i = 1:10:testDBSize
    subplot(2,2,2)
    imshow(testImg(:,:,:,i))
    if bestScore(i) > 0.3
    title([uniqueScenes(best(i)) ' with score ' bestScore(i)])
    end
    drawnow
end
figure;
plot(1:testDBSize,best);
set(gca, 'YTickLabel',uniqueScenes, 'YTick',1:numel(uniqueScenes))