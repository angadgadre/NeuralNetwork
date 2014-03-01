function label = knnPredict(k, train_data, train_label, test_data)
% knnPredict predicts the label of given data by using k-nearest neighbor
% classification algorithm

% Input:
% k: the parameter k of k-nearest neighbor algorithm
% data: matrix of data. Each row of this matrix represents the feature 
%       vector of a particular image

% Output:
% label: a column vector of predicted labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~, I] = pdist2(train_data, test_data,'cosine','Smallest',k);
size_test = size(test_data, 1);
size_train = size(train_data, 1);
test_label = train_label(sub2ind([size_train k], I, ones(k, size_test)), 1);
predicted_labels = reshape(test_label, k, size_test);
label = mode(predicted_labels, 1)';
end

