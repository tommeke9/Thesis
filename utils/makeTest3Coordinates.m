% clear all
% close all
% clc
function [ ImageCoordinates ] = makeTest3Coordinates(  )
ImageCoordinates = zeros(1991,2);

%Room 91.56
ImageCoordinates(1,:) = [490,289];

%Room 91.59
ImageCoordinates(556,:) = [490,388];

%Room 91.88
ImageCoordinates(731,:) = [490,432];

%Room 91.61
ImageCoordinates(967,:) = [490,446];

%Room 91.68
ImageCoordinates(1106,:) = [492,477];

%Room 91.69
%skipped

%Room 91.70
%skipped

%Room 91.71
ImageCoordinates(1455,:) = [407,477];

%End of corridor
ImageCoordinates(1817,:) = [332,477];
ImageCoordinates(1991,:) = [332,543];

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