clear all
close all
clc

ImageCoordinates = zeros(12681,2);

%Room 91.56
ImageCoordinates(1,:) = [490,289];
ImageCoordinates(140,:) = [490,308];
ImageCoordinates(800,:) = [580,308];
ImageCoordinates(1200,:) = [580,366];
ImageCoordinates(1397,:) = [558,366];
ImageCoordinates(1780,:) = [558,310];
ImageCoordinates(2300,:) = [490,308];

%Room 91.59
ImageCoordinates(2830,:) = [490,388];
ImageCoordinates(3420,:) = [580,400];
ImageCoordinates(3585,:) = [574,416];
ImageCoordinates(3830,:) = [349,399];
ImageCoordinates(4200,:) = [523,422];
ImageCoordinates(4490,:) = [519,388];
ImageCoordinates(4623,:) = [490,388];

%Room 91.88
ImageCoordinates(4780,:) = [490,432];
ImageCoordinates(4900,:) = [464,432];
ImageCoordinates(4990,:) = [464,405];
ImageCoordinates(5400,:) = [405,405];
ImageCoordinates(5650,:) = [405,448];
ImageCoordinates(6000,:) = [464,448];
ImageCoordinates(6100,:) = [464,432];
ImageCoordinates(6235,:) = [490,432];

%Room 91.61
ImageCoordinates(6450,:) = [490,446];
ImageCoordinates(7080,:) = [580,446];
ImageCoordinates(7695,:) = [490,446];

%Room 91.68
ImageCoordinates(7886,:) = [492,477];
ImageCoordinates(8274,:) = [492,543];
ImageCoordinates(8610,:) = [492,477];

%Room 91.69
ImageCoordinates(8720,:) = [466,477];
ImageCoordinates(9000,:) = [466,543];
ImageCoordinates(9400,:) = [466,477];

%Room 91.70
ImageCoordinates(9650,:) = [437,477];
ImageCoordinates(10050,:) = [437,543];
ImageCoordinates(10620,:) = [437,477];

%Room 91.71
ImageCoordinates(10800,:) = [407,477];
ImageCoordinates(11250,:) = [407,543];
ImageCoordinates(11580,:) = [407,477];

%End of corridor
ImageCoordinates(11950,:) = [332,477];
ImageCoordinates(12290,:) = [332,543];
ImageCoordinates(12681,:) = [313,543];

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




