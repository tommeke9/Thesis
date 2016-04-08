%function [  ] = MotionFlow( )
%Output: Speed & Rotation?
%Input: 2 images

%-------------Only in development---------------------
clear all
close all
clc
disp('loading ESAT test DB')
T = load('test.mat');
testImg = T.img;
clear T
testDBSize = size(testImg,4);

ignoreMiddle = 0.5; %percentage of image-width/height in the middle to ignore
activatePlots = 1; %1 if you want to see movie/plots
%-----------------------------------------------------
opticFlow = opticalFlowLK('NoiseThreshold',0.001);
widthIm = size(testImg,2);
widthImHalf = round(widthIm/2);
heightIm = size(testImg,1);
heightImHalf = round(heightIm/2);

widthPixelsToIgnore = round((ignoreMiddle * widthIm)/2); %One sided!
heightPixelsToIgnore = round((ignoreMiddle * heightIm)/2);
for index = 1:testDBSize
    frameGray = rgb2gray(testImg(:,:,:,index));

    flow = estimateFlow(opticFlow,frameGray);
    leftSpeed = sum(sum(flow.Vx(:,1:widthImHalf-widthPixelsToIgnore)))/sum(sum(flow.Vx(:,1:widthImHalf-widthPixelsToIgnore)~=0));%(widthImHalf*heightIm);
    rightSpeed = sum(sum(flow.Vx(:,widthPixelsToIgnore+widthImHalf+1:end)))/sum(sum(flow.Vx(:,widthPixelsToIgnore+widthImHalf+1:end)~=0));%/(widthImHalf*heightIm);
    upSpeed = sum(sum(flow.Vy(1:heightImHalf-heightPixelsToIgnore,:)))/sum(sum(flow.Vx(1:heightImHalf-heightPixelsToIgnore,:)~=0));%/(widthIm*heightImHalf);
    downSpeed = sum(sum(flow.Vy(heightPixelsToIgnore+heightImHalf+1:end,:)))/sum(sum(flow.Vx(heightPixelsToIgnore+heightImHalf+1:end,:)~=0));%/(widthIm*heightImHalf);
    
    leftSpeedIsRight = leftSpeed>0;
    rightSpeedIsRight = rightSpeed>0;
    upSpeedIsDown = upSpeed>0;
    downSpeedIsDown = downSpeed>0;
    if activatePlots
        delete(findall(gcf,'Tag','arrow'))
        imshow(testImg(:,:,:,index))
        hold on
        plot(flow,'DecimationFactor',[5 5],'ScaleFactor',10)
        
        %Plot the unused frame.
        rectangle('Position',[widthImHalf-widthPixelsToIgnore heightImHalf-heightPixelsToIgnore widthPixelsToIgnore*2 heightPixelsToIgnore*2])
        
        %4 arrows for the 4 sectors.
        annotation('arrow',[0.25 0.15+0.2*leftSpeedIsRight],[0.5 0.5],'Tag','arrow')
        annotation('arrow',[0.75 0.65+0.2*rightSpeedIsRight],[0.5 0.5],'Tag','arrow')
        annotation('arrow',[0.5 0.5],[0.25 0.35-0.2*downSpeedIsDown],'Tag','arrow')
        annotation('arrow',[0.5 0.5],[0.75 0.85-0.2*upSpeedIsDown],'Tag','arrow')
        
        %CLIP-OFF DUE TO ARROW THAT IS LIMITED TO 1...
        if abs(leftSpeed) > 1
            leftSpeed = -1+2*leftSpeedIsRight;
        end
        if abs(rightSpeed) > 1
            rightSpeed = -1+2*rightSpeedIsRight;
        end
        
        if ~leftSpeedIsRight && rightSpeedIsRight%WALKING FORWARD
            %annotation('arrow',[0.5 0.5],[0.45 0.65],'Tag','arrow','LineWidth',20,'HeadLength',30,'HeadWidth',50)
            annotation('arrow',[0.5 0.5],[0.5 0.5+0.25*(-leftSpeed+rightSpeed)],'Tag','arrow','color','r')
        
        elseif leftSpeedIsRight && rightSpeedIsRight%TURNING LEFT
            annotation('arrow',[0.5 0.5-0.25*(leftSpeed+rightSpeed)],[0.5 0.5],'Tag','arrow','color','r')
        
        elseif ~leftSpeedIsRight && ~rightSpeedIsRight%TURNING Right
            annotation('arrow',[0.5 0.5-0.25*(leftSpeed+rightSpeed)],[0.5 0.5],'Tag','arrow','color','r')
        end
        hold off
        drawnow
    end
end

%end

