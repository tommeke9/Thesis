% clear all
% close all
% clc
function [ TestCoordinates ] = makeTestCoordinates(  )
TestCoordinates = zeros(10557,2);

%Room 91.56
TestCoordinates(1,:) = [490,289];
TestCoordinates(112,:) = [490,308];
TestCoordinates(712,:) = [580,308];
TestCoordinates(1081,:) = [580,366];
TestCoordinates(1258,:) = [558,366];
TestCoordinates(1538,:) = [558,310];
TestCoordinates(1993,:) = [490,308];

%Room 91.59
TestCoordinates(2422,:) = [490,388];
TestCoordinates(2949,:) = [580,400];
TestCoordinates(3062,:) = [574,416];
TestCoordinates(3243,:) = [549,399];
TestCoordinates(3583,:) = [523,422];
TestCoordinates(3904,:) = [519,388];
TestCoordinates(4033,:) = [490,388];

%Room 91.88
TestCoordinates(4188,:) = [490,432];
TestCoordinates(4327,:) = [464,432];
TestCoordinates(4399,:) = [464,405];
TestCoordinates(4768,:) = [405,405];
TestCoordinates(4996,:) = [405,448];
TestCoordinates(5302,:) = [464,448];
TestCoordinates(5428,:) = [464,432];
TestCoordinates(5543,:) = [490,432];

%Room 91.61
TestCoordinates(5704,:) = [490,446];
TestCoordinates(6285,:) = [580,446];
TestCoordinates(6875,:) = [490,446];

%Room 91.68
TestCoordinates(7043,:) = [492,477];
TestCoordinates(7377,:) = [492,543];
TestCoordinates(7681,:) = [492,477];

%Room 91.69
%Skipped

%Room 91.70
TestCoordinates(7962,:) = [437,477];
TestCoordinates(8339,:) = [437,543];
TestCoordinates(8763,:) = [437,477];

%Room 91.71
TestCoordinates(8904,:) = [407,477];
TestCoordinates(9316,:) = [407,543];
TestCoordinates(9648,:) = [407,477];

%End of corridor
TestCoordinates(9955,:) = [332,477];
TestCoordinates(10226,:) = [332,543];
TestCoordinates(10557,:) = [313,543];

%-------------------Interpolate--------------------------------------------
FixedPhotos = find(TestCoordinates(:,1));

for trajectNumber = 1:size(FixedPhotos,1)-1
        PhotoBefore = FixedPhotos(trajectNumber);
        PhotoAfter = FixedPhotos(trajectNumber+1);
        
        PhotoLocBefore = TestCoordinates(PhotoBefore,:);
        PhotoLocAfter = TestCoordinates(PhotoAfter,:);
        AmountToInterpolate = PhotoAfter - PhotoBefore;
        i = 1;
    for index = PhotoBefore+1:PhotoAfter-1 %Every photo to be interpolated
        ImageCoordinatestemp =  PhotoLocBefore + i * (PhotoLocAfter-PhotoLocBefore)/AmountToInterpolate;
        TestCoordinates(index,:) = round(ImageCoordinatestemp);
        i = i+1;
    end
end

end
%save('ImageCoordinates.mat','ImageCoordinates');