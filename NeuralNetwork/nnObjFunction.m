function [obj_val, obj_grad] = nnObjFunction(params, n_input, n_hidden, ...
                                    n_class, training_data,...
                                    training_label, lambda)
% nnObjFunction computes the value of objective function (negative log 
%   likelihood error function with regularization) given the parameters 
%   of Neural Networks, thetraining data, their corresponding training 
%   labels and lambda - regularization hyper-parameter.

% Input:
% params: vector of weights of 2 matrices w1 (weights of connections from
%     input layer to hidden layer) and w2 (weights of connections from
%     hidden layer to output layer) where all of the weights are contained
%     in a single vector.
% n_input: number of node in input layer (not include the bias node)
% n_hidden: number of node in hidden layer (not include the bias node)
% n_class: number of node in output layer (number of classes in
%     classification problem
% training_data: matrix of training data. Each row of this matrix
%     represents the feature vector of a particular image
% training_label: the vector of truth label of training images. Each entry
%     in the vector represents the truth label of its corresponding image.
% lambda: regularization hyper-parameter. This value is used for fixing the
%     overfitting problem.
       
% Output: 
% obj_val: a scalar value representing value of error function
% obj_grad: a SINGLE vector of gradient value of error function
% NOTE: how to compute obj_grad
% Use backpropagation algorithm to compute the gradient of error function
% for each weights in weight matrices.
% Suppose the gradient of w1 is 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reshape 'params' vector into 2 matrices of weight w1 and w2
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit i in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in hidden 
%     layer to unit i in output layer.
w1 = reshape(params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% grad_w2 = zeros(n_class, (n_hidden + 1));
% grad_w1 = zeros(n_hidden, (n_input + 1));

t = zeros((n_input + 1), n_class);
for N = 1 : size(training_data, 1)
    t(N, training_label(N) + 1) = 1;
end

sum_w2 = zeros(n_class, (n_hidden + 1));
sum_w1 = zeros(n_hidden, (n_input + 1));

training_data2 = [training_data, ones(N,1)];
err_fn_n = zeros(n_class, size(training_data, 1));
for N = 1 : size(training_data, 1)
    a = w1 * training_data2(N, :)';
    z = [sigmoid(a); 1];
        
    b = w2 * z;
    y = sigmoid(b);
    
    grad_w2_n = zeros(n_class, (n_hidden + 1));
    delta = y - t(N, :)';
    err_fn_n(:, N) = -(t(N, :)' .* log(y) + (1 - t(N, :)') .* log(1 - y));
    for k = 1 : n_class
        for j = 1:(n_hidden + 1)
            grad_w2_n(k, j) = delta(k) * z(j);
        end
    end
    
    grad_w1_n = zeros(n_hidden, (n_input + 1));
    for j = 1 : n_hidden
        sum_dk_w2kj = sum(delta(:) .* w2(:, j));
        grad_w1_n(j,:) = (1 - z(j)) * z(j) * sum_dk_w2kj * training_data2(N,:);
    end
    
    sum_w2 = sum_w2 + grad_w2_n;
    sum_w1 = sum_w1 + grad_w1_n;
end
grad_w2 = (1/N)*(sum_w2 + lambda*w2);
grad_w1 = (1/N)*(sum_w1 + lambda*w1);

% Suppose the gradient of w1 and w2 are stored in 2 matrices grad_w1 and grad_w2
% Unroll gradients to single column vector
obj_grad = [grad_w1(:) ; grad_w2(:)];

err_fn = sum(sum(err_fn_n, 1), 2);
obj_val = (1/N) * (err_fn + (lambda/2)*(sum(sum(w1.^2,1),2) + sum(sum(w2.^2,1),2)));

end
