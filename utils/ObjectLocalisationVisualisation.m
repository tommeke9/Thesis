% check setup instructions in readme
clear all
close all
clc

addpath data deps/matconvnet-1.0-beta16 data/ESAT-DB

disp('loading ESAT DB')
T = load('test.mat');
testImg = T.img;
clear T
disp('DB loaded')

% Define the sizes of the DB
testDBSize = size(testImg,4);

%%%%%%%%%%%%%%%%%%%%%
%set parameters and load models
%%%%%%%%%%%%%%%%%%%%%
n_box_max = 5;



%add necessary paths
addpath('./deps/edges-master/');
addpath(genpath('./deps/piotr_toolbox_V3.40/toolbox/'));
run('./deps/matconvnet-1.0-beta16/matlab/vl_setupnn');

%models
opts.model.contour = './data/deepproposal/contour/modelF_C2.mat';
opts.model.cnn = './data/deepproposal/cnn/imagenet-caffe-ref.mat';
opts.model.objectness = './data/deepproposal/objectness/';

%set parameters
opts.thr_s = 0.5; 
opts.thr_e = opts.thr_s; %if using adaptive nms introduced in "What makes for effective detection proposals?", arXiv 2015.
opts.nbox_s1 = 4000; %%6000;
opts.nbox_s2 = 3000; %4000;
opts.nbox_s3 = 1000;
opts.layers = [14 10 6];
opts.scales = [227 300 400 600];
opts.nsliding_win = 50;
opts.step_siz = (opts.thr_s/opts.thr_e).^(1/opts.nbox_s3);

%load windows sizes obtained based on algorithm explained in sec3.1 of the paper
X = load('best_siz_win_AR_VOC07.mat','siz_win');
win_sizes_i = X.siz_win(1:opts.nsliding_win);
for i=1:length(opts.scales)
    win_sizes{i} = win_sizes_i;
end

%cnn
net = load(opts.model.cnn);
net_gpu = net;%vl_simplenn_move(net, 'gpu'); %Work with CPU for now

%load objectness models
mdl_obj = train_objectness(win_sizes, net_gpu, opts);

%load contour model
mdl_contour = load(opts.model.contour); 
mdl_contour = mdl_contour.model;


%%%%%%%%%%%%%%%%%%%%%
%main process
%%%%%%%%%%%%%%%%%%%%%
fprintf('Calculating object locations');
tic
for i = 1:testDBSize
    %compute feature maps
    x_map = compute_featmaps(testImg(:,:,:,i), net_gpu, opts);

    %entry to the deepProposal
    boxes = deepProposal( testImg(:,:,:,i), x_map, mdl_obj, mdl_contour, win_sizes, opts );
    nbox = size(boxes, 1);
    testObjectLocation(:,:,i)=boxes(1:min(n_box_max,nbox), :); 
    if rem(i,100)==0
            fprintf('image %d ~ %d of %d \n',i-99,i,testDBSize);
    end
end
toc
if exist('data/Objects.mat', 'file')
  save('data/Objects.mat','testObjectLocation','-append');
else
  save('data/Objects.mat','testObjectLocation');
end
fprintf('ObjectLocations are saved\n');
%%%%%%%%%%%%%%%%%%%%%
%visualization
%%%%%%%%%%%%%%%%%%%%%
figure('units','normalized','outerposition',[0 0 1 1]);
v = VideoWriter('data/ObjectLocalisation.avi');
open(v)
for i = 1:testDBSize
    testObjectLocation(:,[3 4],i)= testObjectLocation(:,[3 4],i)-testObjectLocation(:,[1 2],i)+1;
    %visualization of proposals    
    
    %show first n_box_show boxes in the image
    subplot(1,2,1)
    imshow(testImg(:,:,:,i));
    colors=[];
    for r=1:size(testObjectLocation,1)
        colors=cat(1,colors, rand(1,3)); 
        rectangle('Position', testObjectLocation(r,[1:4],i), 'EdgeColor', colors(r,:), 'LineWidth', 3); 
    end
    title(['Object proposal of test image: ',num2str(i)])

    %heatmap of boxes
    im_heat = zeros(size(testImg(:,:,:,i),1), size(testImg(:,:,:,i),2)); 
    %testImg(:,:,:,i)=double(testImg(:,:,:,i));
    for o=1:n_box_max %size(boxes_i,1)
        bb=testObjectLocation(o,[1:4],i);
        im_heat(bb(2):bb(4), bb(1):bb(3)) = im_heat(bb(2):bb(4), bb(1):bb(3)) + single(testImg(bb(2):bb(4), bb(1):bb(3),1,i));
    end
    subplot(1,2,2), imshow(im_heat,[]);
     title('Heatmap of the object locations')
    frame = getframe(gcf);
     writeVideo(v,frame)
end
close(v);