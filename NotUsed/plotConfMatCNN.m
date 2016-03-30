figure;
undex = 1;
for ConfMatCNN = [0.5,0.625,0.75,0.875,1]

%------------------------Combine Confusion Matrices------------------------
disp('Start combining the confusion matrices')

%confusionMatrix = ConfMatCNN .* confusionMatrixCNNFeat + (1-ConfMatCNN) .* confusionMatrixSceneRecogn;
%confusionMatrix = confusionMatrixCNNFeat .* confusionMatrixSceneRecogn;
confusionMatrix = ConfMatCNN .* (confusionMatrixCNNFeat - min(min(confusionMatrixCNNFeat)))./max(max(confusionMatrixCNNFeat)) + (1-ConfMatCNN) .* (confusionMatrixSceneRecogn - min(min(confusionMatrixSceneRecogn)))./max(max(confusionMatrixSceneRecogn));
subplot(2,5,5+undex)
imagesc(confusionMatrix)
title({['Combined Confusion Matrix'],['with ConfMatCNN=' num2str(ConfMatCNN)]})
xlabel('Training Image')
ylabel('Test Image')


%--------------------------------------------------------------------------

%------------------------Select Lowest difference--------------------------
disp('Search lowest difference')
parfor index = 1:testDBSize
    [ResultValue(index),Result(index)] = min(confusionMatrix(:,index));
%     if rem(index,100)==0
%             fprintf('%d ~ %d of %d \n',index-99,index,testDBSize);
%     end
end
subplot(2,5,undex);
plot(Result,'g')
hold on

%------------------------Spatial Continuity check--------------------------
d = 2; % Length of evaluation window
epsilon = 3;
for index = d:testDBSize
    P(index) = 1;
    for u = index-d+2:index
        if abs(ResultValue(u-1)-ResultValue(u)) > epsilon
            P(index) = 0;
            break;
        end
    end
end

%HOLD THE PREVIOUS VALUE IF P=0
Resultnew(1) = Result(1);
for index = 2:testDBSize
    if P(index) == 1
        Resultnew(index) = Result(index);
    else
        Resultnew(index) = Resultnew(index-1);
    end
end
plot(Resultnew,'r')
%plot(Resultnew-Result,'g')
hold off
title({['Green = initial, Red = after SC check with:'],['epsilon=' num2str(epsilon) '; d=' num2str(d) '; ConfMatCNN=' num2str(ConfMatCNN)]})

undex = undex+1;
end