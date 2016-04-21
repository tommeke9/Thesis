% clear all
% close all
% clc
function [ ImageCoordinates ] = makeTest4Coordinates(  )
ImageCoordinates = zeros(2835,2);

%Room 91.56
ImageCoordinates(1,:) = [490,308];

%Room 91.59
ImageCoordinates(469,:) = [490,388];

%Room 91.88
ImageCoordinates(675,:) = [490,432];
ImageCoordinates(794,:) = [464,432];
ImageCoordinates(862,:) = [464,405];

ImageCoordinates(965,:) = [464,405];
ImageCoordinates(1005,:) = [464,432];
ImageCoordinates(1131,:) = [490,432];

%Room 91.61
ImageCoordinates(1270,:) = [490,446];

%Room 91.68
ImageCoordinates(1455,:) = [492,477];

%Room 91.69
ImageCoordinates(1500,:) = [466,477];

ImageCoordinates(2500,:) = [466,477];

%Room 91.70
%skipped

%Room 91.71
ImageCoordinates(2680,:) = [407,477];

ImageCoordinates(2835,:) = [407,477];

%End of corridor
%skipped

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