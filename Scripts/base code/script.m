clearvars;

[train_data, train_label, validation_data, ...
    validation_label, test_data, test_label] = preprocess();

save('dataset.mat', 'train_data', 'train_label', 'validation_data', ...
                    'validation_label', 'test_data', 'test_label');
load('dataset.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% **************Neural Network********************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Train Neural Network

% set the number of nodes in input unit (not including bias unit)
n_input = size(train_data, 2); 
% set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;				   
% set the number of nodes in output unit
n_class = 10;				   

% initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

% unroll 2 weight matrices into single column vector
initialWeights = [initial_w1(:); initial_w2(:)];

% set the maximum number of iteration in conjugate gradient descent
options = optimset('MaxIter', 50);

% set the regularization hyper-parameter
lambda = 0;

% define the objective function
objFunction = @(params) nnObjFunction(params, n_input, n_hidden, ...
                       n_class, train_data, train_label, lambda);

% run neural network training with fmincg
[nn_params, cost] = fmincg(objFunction, initialWeights, options);

% reshape the nn_params from a column vector into 2 matrices w1 and w2
w1 = reshape(nn_params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(nn_params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));

%   Test the computed parameters
predicted_label = nnPredict(w1, w2, train_data);
fprintf('\nTraining Set Accuracy: %f\n', ...
         mean(double(predicted_label == train_label)) * 100);

%   Test Neural Network with validation data
predicted_label = nnPredict(w1, w2, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', ...
         mean(double(predicted_label == validation_label)) * 100);

%   Test Neural Network with test data
predicted_label = nnPredict(w1, w2, test_data);
fprintf('\nTesting Set Accuracy: %f\n', ...
         mean(double(predicted_label == test_label)) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% **************K-Nearest Neighbors***************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 1;
%   Test KNN with validation data
predicted_label = knnPredict(k, train_data, train_label, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', ...
         mean(double(predicted_label == validation_label)) * 100);

%   Test KNN with test data
predicted_label = knnPredict(k, train_data, train_label, test_data);
fprintf('\nTesting Set Accuracy: %f\n', ...
         mean(double(predicted_label == test_label)) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% *******Save the learned parameters *************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('params.mat', 'n_input', 'n_hidden', 'w1', 'w2', 'lambda', 'k');