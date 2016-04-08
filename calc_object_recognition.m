function [ scores ] = calc_object_recognition( images, boxes, net, n_max_labels )
%calc_object_recognition: calculate the scores for every object inside the
%given boxes.
%OUTPUT: scores = (label for the 5 most possible objects, scores for these objects , box number , Image number)
%Labels & scores are sorted from high to low

delete(gcp('nocreate'))
addpath data deps/matconvnet-1.0-beta16 data/ESAT-DB

% Define the sizes of the DB
DBSize = size(images,4);
n_boxes = size(boxes,1);
run('./deps/matconvnet-1.0-beta16/matlab/vl_setupnn');

if sum(size(net.meta.normalization.averageImage)) == 4
    averageImage(:,:,1) = net.meta.normalization.averageImage(1) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,2) = net.meta.normalization.averageImage(2) * ones(net.meta.normalization.imageSize(1:2));
    averageImage(:,:,3) = net.meta.normalization.averageImage(3) * ones(net.meta.normalization.imageSize(1:2));
else
    averageImage = net.meta.normalization.averageImage;
end

n_labels = min(n_max_labels,size(net.meta.classes.name,2));

scores = zeros(n_labels,2,n_boxes,DBSize);
for index = 1:DBSize
    bb = boxes(:,1:4,index);
    %bb(:,[3 4])= bb(:,[3 4])-bb(:,[1 2])+1;
    for i = 1:n_boxes
        %resize the box
        im_temp = images(bb(i,2):bb(i,4),bb(i,1):bb(i,3),:,index);
        im_temp = imresize(im_temp, net.meta.normalization.imageSize(1:2)) ;
        im_temp = single(im_temp) - averageImage ;
        
        % run the CNN
        res = vl_simplenn(net, im_temp) ;
        scores_temp = squeeze(gather(res(end).x));
        
        [val, ind] = sort(scores_temp,'descend') ;
        scores(:,:,i,index) = [ind(1:n_labels), val(1:n_labels)];
    end
    if rem(index,100)==0
        fprintf('image %d ~ %d of %d analyzed for objects\n',index-99,index,DBSize);
    end
end
end

