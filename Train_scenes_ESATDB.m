function [ scores ] = Train_scenes_ESATDB( images,net,lastFClayer )
%TRAIN_SCENES_ESATDB Train the ESATDB training OR test Database for scenes
%GOAL: Save for every image the scores for every scenetype. (Thus
%a score for every scene in the DB). 
%Function is only called if retraining is necessary.

DBSize = size(images,4);

%Setup MatConvNet
run matconvnet-1.0-beta16/matlab/vl_setupnn;

load('ESATsvm.mat');

%imageNorm = zeros([size(net.normalization.averageImage) DBSize]);
%scores = zeros(DBSize,size(uniqueScenes,1));

for index = 1:DBSize
        %normalize
        im_temp = single(images(:,:,:,index)) ; % note: 0-255 range
        im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
        imageNorm(:,:,:,index) = im_temp - net.normalization.averageImage ;

        %Run CNN
        res = vl_simplenn(net, imageNorm(:,:,:,index)) ;
        lastFCtemp = squeeze(gather(res(lastFClayer+1).x));
        lastFCTest = lastFCtemp(:);

        %Select correct scene
        for i = 1:size(uniqueScenes,1)
            scores(index,i) = W(:,i)'*lastFCTest + B(i) ;
        end
end
end

