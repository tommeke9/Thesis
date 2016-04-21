% clear all
% close all
% clc
function [ ImageCoordinates ] = makeTest5Coordinates(  ) %DIFFERENT DIRECTION
ImageCoordinates = zeros(1822,2);


%Room 91.56
ImageCoordinates(1822,:) = [490,308];


%Room 91.59


%Room 91.88
ImageCoordinates(1383,:) = [490,432];

%Room 91.61

%Room 91.68
ImageCoordinates(1171,:) = [492,477];

%Room 91.69
ImageCoordinates(991,:) = [466,477];

%Room 91.70


%Room 91.71


%End of corridor
ImageCoordinates(392,:) = [332,477];
ImageCoordinates(1,:) = [332,543];

%-------------------Interpolate--------------------------------------------
FixedPhotos = find(ImageCoordinates(:,1));

for trajectNumber = 1:size(FixedPhotos,1)-1
        PhotoBefore = FixedPhotos(trajectNumber);
        PhotoAfter = FixedPhotos(trajectNumber+1);
        
        PhotoLocBefore = ImageCoordinates(PhotoBefore,:);
        PhotoLocAfter = ImageCoordinates(PhotoAfter,:);
        AmountToInterpolate = PhotoAfter - PhotoBefore;
        i = 1;
    for index = PhotoBefore+1:PhotoAfter-1 %Every photo to be interpolated
        ImageCoordinatestemp =  PhotoLocBefore + i * (PhotoLocAfter-PhotoLocBefore)/AmountToInterpolate;
        ImageCoordinates(index,:) = round(ImageCoordinatestemp);
        i = i+1;
    end
end

end
%save('ImageCoordinates.mat','ImageCoordinates');