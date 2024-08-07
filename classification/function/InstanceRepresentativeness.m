function [Q, representativenessRank] = InstanceRepresentativeness(X, paraDc)
[num_instance, ~] = size(X);
tempDistanceMatrix = pdist2(X, X,'euclidean');
instanceSimilarMatrix = exp(-tempDistanceMatrix.^2./ paraDc^2);
tempDensityArray = sum(instanceSimilarMatrix);

numK = 15;
dist_max = diag(realmax*ones(1,num_instance));
dist = tempDistanceMatrix + dist_max;
indicator_position = zeros(size(dist));

tempDistanceToMasterArray = zeros(1, num_instance);
for i = 1:num_instance
    tempIndex = tempDensityArray>tempDensityArray(i);
    if sum(tempIndex) > 0
        tempDistanceToMasterArray(1,i) = min(tempDistanceMatrix(i,tempIndex)); 
    end
    
    [~,index] = sort(dist(i,:));
    label_neighbor_index = index(1:numK);
    indicator_position(i,label_neighbor_index) = 1;
end

Q = indicator_position.*instanceSimilarMatrix;

representativenessArray = tempDensityArray.*tempDistanceToMasterArray;
[~, representativenessRank] = sort(representativenessArray, 'descend');
end