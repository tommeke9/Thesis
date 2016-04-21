% clear all
% close all
% clc
function [ ImageCoordinates ] = makeTest6Coordinates(  )
ImageCoordinates = zeros(1134,2);

%Room 91.56

%Room 91.59

%Room 91.88

%Room 91.61
ImageCoordinates(1,:) = [490,446];

%Room 91.68
ImageCoordinates(120,:) = [492,477];

%Room 91.69

%Room 91.70

%Room 91.71
ImageCoordinates(415,:) = [407,477];

%End of corridor
ImageCoordinates(844,:) = [332,477];

ImageCoordinates(1022,:) = [332,477];

ImageCoordinates(1134,:) = [370,477];

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