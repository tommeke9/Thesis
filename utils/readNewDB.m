clear all
close all
clc

addpath data/newDB

disp('loading dataset')
folderInfo = dir('data/newDB');
folderInfo = folderInfo(arrayfun(@(x) x.name(1), folderInfo) ~= '.');

number = 1;
for i = 1:size(folderInfo,1)
    imageInfo = dir(['data/newDB/' folderInfo(i).name]);
    imageInfo = imageInfo(arrayfun(@(x) x.name(1), imageInfo) ~= '.');
    
    for index = 1:size(imageInfo,1)
        im_temp = imread(fullfile(['data/newDB/' folderInfo(i).name],imageInfo(index).name));
        if size(im_temp,3) ~=1 %Ignore B&W images    
            images(:,:,:,number) = imresize(im_temp,[256 144]);
            sceneTypes{number,1} = folderInfo(i).name;
            number = number + 1;
        end
        if rem(index,100)==0
            fprintf('%d ~ %d of %s \n',index-99,index,folderInfo(i).name);
        end
    end
    disp([folderInfo(i).name ' loaded'])
end

save('data/newDB.mat','images','sceneTypes')
