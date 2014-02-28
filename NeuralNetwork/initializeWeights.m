function W = initializeWeights(n_in, n_out)
% initializeWeights return the random weights for Neural Network given the
% number of node in the input layer and output layer

% Input:
% n_in: number of nodes of the input layer
% n_out: number of nodes of the output layer
       
% Output: 
% W: matrix of random initial weights with size (n_out x (n_in + 1))
epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
W = rand(n_out, n_in + 1) * 2* epsilon - epsilon;

end

