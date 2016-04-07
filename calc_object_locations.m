function [ ObjectLocation ] = calc_object_locations( n_box_max, images )
%Calculate the object locations using the DeepProposals algortihm from 'A.
%Gohdrati et Al'
%OUTPUT: ObjectLocation(n_box_max,5,ImgDbSize) ==> each box has this format: [x y x+w y+h]

addpath data deps/matconvnet-1.0-beta16 data/ESAT-DB

% Define the sizes of the DB
DBSize = size(images,4);

%%%%%%%%%%%%%%%%%%%%%
%set parameters and load models
%%%%%%%%%%%%%%%%%%%%%
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
ObjectLocation = zeros(n_box_max,5,DBSize);
for i = 1:DBSize
    %compute feature maps
    x_map = compute_featmaps(images(:,:,:,i), net_gpu, opts);

    %entry to the deepProposal
    boxes = deepProposal( images(:,:,:,i), x_map, mdl_obj, mdl_contour, win_sizes, opts );
    nbox = size(boxes, 1);
    ObjectLocation(:,:,i)=boxes(1:min(n_box_max,nbox), :);
end
end

