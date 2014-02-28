function label = nnPredict(w1, w2, data)
% nnPredict predicts the label of data given the parameter w1, w2 of Neural
% Network.

% Input:
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% data: matrix of data. Each row of this matrix represents the feature 
%       vector of a particular image
       
% Output: 
% label: a column vector of predicted labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     label = zeros(size(data, 1), 1);
    bias_data = [data, ones(size(data, 1), 1)];
    z = [w1 * bias_data'; ones(1, size(data, 1))];
    y = w2 * z;
    [~,label] = max(y);
    label = label - 1;
    label = label';
%     for i=1:size(data, 1)
%         label(i) = find(y(:,i)'>0) - 1;
%     end
end
