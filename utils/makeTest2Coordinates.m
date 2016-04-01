% clear all
% close all
% clc
function [ Test2Coordinates ] = makeTest2Coordinates(  )
Test2Coordinates = zeros(9143,2);

%Room 91.56
Test2Coordinates(1,:) = [490,289];
%skipped

%Room 91.59
Test2Coordinates(888,:) = [490,388];
Test2Coordinates(1579,:) = [580,400];
Test2Coordinates(1700,:) = [574,416];
Test2Coordinates(1977,:) = [549,399];
Test2Coordinates(2315,:) = [519,388];
Test2Coordinates(2494,:) = [490,388];

%Room 91.88
Test2Coordinates(2666,:) = [490,432];
Test2Coordinates(2849,:) = [464,432];
Test2Coordinates(2891,:) = [464,405];
Test2Coordinates(3273,:) = [405,405];
Test2Coordinates(3521,:) = [405,448];
Test2Coordinates(3779,:) = [464,448];
Test2Coordinates(3911,:) = [464,432];
Test2Coordinates(4074,:) = [490,432];

%Room 91.61
Test2Coordinates(4288,:) = [490,446];
Test2Coordinates(5068,:) = [580,446];
Test2Coordinates(5739,:) = [490,446];

%Room 91.68
Test2Coordinates(5879,:) = [492,477];
Test2Coordinates(6212,:) = [492,543];
Test2Coordinates(6598,:) = [492,477];

%Room 91.69
%skipped

%Room 91.70
Test2Coordinates(6855,:) = [437,477];
Test2Coordinates(7211,:) = [437,543];
Test2Coordinates(7532,:) = [437,477];

%Room 91.71
Test2Coordinates(7651,:) = [407,477];
Test2Coordinates(7975,:) = [407,543];
Test2Coordinates(8315,:) = [407,477];

%End of corridor
Test2Coordinates(8638,:) = [332,477];
Test2Coordinates(8968,:) = [332,543];
Test2Coordinates(9143,:) = [313,543];

%-------------------Interpolate--------------------------------------------
FixedPhotos = find(Test2Coordinates(:,1));

for trajectNumber = 1:size(FixedPhotos,1)-1
        PhotoBefore = FixedPhotos(trajectNumber);
        PhotoAfter = FixedPhotos(trajectNumber+1);
        
        PhotoLocBefore = Test2Coordinates(PhotoBefore,:);
        PhotoLocAfter = Test2Coordinates(PhotoAfter,:);
        AmountToInterpolate = PhotoAfter - PhotoBefore;
        i = 1;
    for index = PhotoBefore+1:PhotoAfter-1 %Every photo to be interpolated
        ImageCoordinatestemp =  PhotoLocBefore + i * (PhotoLocAfter-PhotoLocBefore)/AmountToInterpolate;
        Test2Coordinates(index,:) = round(ImageCoordinatestemp);
        i = i+1;
    end
end

end
%save('ImageCoordinates.mat','ImageCoordinates');