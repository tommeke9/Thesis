function [ scores ] = calc_object_recognition( images, boxes, net )
%calc_object_recognition: calculate the scores for every object inside the
%given boxes.
%OUTPUT: scores = (scores for all the possible objects , box number , Image number)
delete(gcp('nocreate'))
addpath data deps/matconvnet-1.0-beta16 data/ESAT-DB

% Define the sizes of the DB
DBSize = size(images,4);
n_boxes = size(boxes,1);
run('./deps/matconvnet-1.0-beta16/matlab/vl_setupnn');

scores = zeros(size(net.classes.name,2),n_boxes,DBSize);
for index = 1:DBSize
    bb = boxes(:,1:4,index);
    %bb(:,[3 4])= bb(:,[3 4])-bb(:,[1 2])+1;
    for i = 1:n_boxes
        %resize the box
        im_temp = images(bb(i,2):bb(i,4),bb(i,1):bb(i,3),:,index);
        im_temp = imresize(im_temp, net.normalization.imageSize(1:2)) ;
        im_temp = single(im_temp) - net.normalization.averageImage ;
        
        % run the CNN
        res = vl_simplenn(net, im_temp) ;
        scores(:,i,index) = squeeze(gather(res(end).x));
    end
    if rem(index,100)==0
        fprintf('image %d ~ %d of %d \n',index-99,index,DBSize);
    end
end
end

