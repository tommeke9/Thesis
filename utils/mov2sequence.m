clear all
close all
clc

addpath utils data/ESAT-DB
disp('loading movie')
inputVideo = VideoReader('Test6.mov');
disp('converting movie')

i = 1;
%img = zeros(144,256,3,12681);
while hasFrame(inputVideo)
   tempImg = readFrame(inputVideo);
   img(:,:,:,i) = imresize(tempImg,[NaN 256]);
   i = i+1;
end

disp('Writing images')
for i = 1:size(img,4)
    imwrite(img(:,:,:,i),fullfile('data/ESAT-DB/original/test6',strcat('im',num2str(i),'.jpg')));
end

save('data/ESAT-DB/mat/test6.mat','img','-v7.3');
%29.9821*422.9517 frames = 12681